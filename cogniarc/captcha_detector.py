"""CAPTCHA detection + classification pipeline.
Finds CAPTCHA elements on a page via CSS selectors, crops them, and classifies.

Supports: reCAPTCHA v2, hCaptcha, Cloudflare Turnstile, text, math.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List


# CSS selectors for common CAPTCHA types
CAPTCHA_SELECTORS = {
    "recaptcha_v2": [
        'iframe[src*="recaptcha/api2/anchor"]',
        'iframe[src*="recaptcha/api2/bframe"]',
    ],
    "hcaptcha": [
        'iframe[src*="hcaptcha.com/captcha"]',
        'iframe[src*="hcaptcha.com/checkbox"]',
    ],
    "turnstile": [
        'iframe[src*="challenges.cloudflare.com/turnstile"]',
        'div.cf-turnstile',
    ],
    "text_captcha": [
        'img[src*="captcha"]',
        'img[id*="captcha"]',
        'input[name*="captcha"]',
    ],
}


class CaptchaDetector:
    """Detect and classify CAPTCHAs on web pages using CSS selectors."""
    
    def __init__(self):
        self._last_result = None
    
    def detect_in_browser(self, browser_console_fn) -> Dict:
        """Detect CAPTCHAs in the current browser page.
        
        Args:
            browser_console_fn: function that evaluates JS in browser context
                Signature: fn(expression: str) -> dict with 'result' key
        
        Returns:
            {
                'found': bool,
                'type': str or None,
                'selector': str or None,
                'bounding_box': {x, y, width, height} or None,
                'confidence': float or None,
            }
        """
        # Try each CSS selector
        for cap_type, selectors in CAPTCHA_SELECTORS.items():
            for selector in selectors:
                js = f"""
                (function() {{
                    const el = document.querySelector('{selector}');
                    if (!el) return null;
                    const rect = el.getBoundingClientRect();
                    return {{
                        x: rect.x, y: rect.y,
                        width: rect.width, height: rect.height,
                        visible: rect.width > 0 && rect.height > 0
                    }};
                }})()
                """
                result = browser_console_fn(js)
                
                if result and result.get('visible'):
                    return {
                        'found': True,
                        'type': cap_type,
                        'selector': selector,
                        'bounding_box': {
                            'x': int(result['x']),
                            'y': int(result['y']),
                            'width': int(result['width']),
                            'height': int(result['height']),
                        },
                        'confidence': 0.9,  # CSS match = high confidence
                    }
        
        # Check for generic CAPTCHA indicators
        js = """
        (function() {
            const indicators = [
                document.querySelector('iframe[src*="captcha"]'),
                document.querySelector('iframe[src*="challenge"]'),
                document.querySelector('.g-recaptcha'),
                document.querySelector('[data-sitekey]'),
            ].filter(Boolean);
            return indicators.length > 0 ? {count: indicators.length} : null;
        })()
        """
        result = browser_console_fn(js)
        if result and result.get('count', 0) > 0:
            return {
                'found': True,
                'type': 'unknown',
                'selector': None,
                'bounding_box': None,
                'confidence': 0.5,
            }
        
        return {'found': False, 'type': None, 'selector': None, 'bounding_box': None, 'confidence': None}
    
    def detect_in_html(self, html: str) -> List[Dict]:
        """Detect CAPTCHAs in raw HTML (offline).
        
        Returns list of detections.
        """
        detections = []
        
        for cap_type, selectors in CAPTCHA_SELECTORS.items():
            for selector in selectors:
                if selector.split('[')[0] in html.lower():
                    detections.append({
                        'type': cap_type,
                        'selector': selector,
                        'confidence': 0.7,  # HTML match = medium confidence
                    })
                    break  # One match per type
        
        return detections
    
    def classify_crop(self, screenshot: np.ndarray, box: Dict,
                     classifier=None) -> Tuple[str, float]:
        """Crop a region from screenshot and classify it.
        
        Args:
            screenshot: full page screenshot as numpy array
            box: {'x', 'y', 'width', 'height'}
            classifier: CaptchaPredictor instance
        
        Returns:
            (type_name, confidence)
        """
        if classifier is None:
            from cogniarc.micro_predictors import CaptchaPredictor
            classifier = CaptchaPredictor()
        
        if not classifier.available:
            return 'unknown', 0.0
        
        x, y, w, h = int(box['x']), int(box['y']), int(box['width']), int(box['height'])
        
        # Clamp to image bounds
        sh, sw = screenshot.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, sw - x), min(h, sh - y)
        
        if w <= 0 or h <= 0:
            return 'none', 0.0
        
        # Crop
        crop = screenshot[y:y+h, x:x+w]
        return classifier.classify_screenshot(crop)
    
    def full_pipeline(self, screenshot: np.ndarray,
                      browser_console_fn=None,
                      classifier=None) -> Dict:
        """Run full detection + classification pipeline.
        
        1. Detect CAPTCHA element via CSS (if browser context available)
        2. Crop the region from screenshot
        3. Classify with micro-NN
        
        Returns:
            {
                'captcha_detected': bool,
                'css_type': str or None,
                'nn_type': str or None,
                'nn_confidence': float or None,
                'bounding_box': dict or None,
            }
        """
        result = {
            'captcha_detected': False,
            'css_type': None,
            'nn_type': None,
            'nn_confidence': None,
            'bounding_box': None,
        }
        
        # Step 1: CSS detection
        if browser_console_fn:
            detection = self.detect_in_browser(browser_console_fn)
            if detection['found'] and detection['bounding_box']:
                result['css_type'] = detection['type']
                result['bounding_box'] = detection['bounding_box']
                result['captcha_detected'] = True
        
        # Step 2+3: Crop + classify
        if result['bounding_box']:
            nn_type, nn_conf = self.classify_crop(
                screenshot, result['bounding_box'], classifier
            )
            result['nn_type'] = nn_type
            result['nn_confidence'] = nn_conf
            
            # If NN disagrees with CSS, trust NN (it sees the actual pixels)
            if nn_conf > 0.8 and nn_type != 'none':
                result['captcha_detected'] = True
        
        return result
