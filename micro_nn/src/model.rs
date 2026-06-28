use serde::Deserialize;
use crate::activation::Activation;

/// Neural network weights format (botte-secrete compatible).
/// Weights are flat 1D arrays: weights[i] = output_size * input_size (row-major).
#[derive(Debug, Deserialize)]
pub struct Weights {
    pub layers: Vec<usize>,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<Vec<f64>>,
    pub activations: Vec<Activation>,
    #[serde(default)]
    pub labels: Vec<String>,
}

/// Tiny feedforward neural network — inference only.
pub struct Model {
    weights: Weights,
}

impl Model {
    /// Load model from JSON string (botte-secrete format).
    pub fn from_json(json: &str) -> Result<Self, String> {
        let w: Weights = serde_json::from_str(json)
            .map_err(|e| format!("Failed to parse model JSON: {}", e))?;
        
        // Validate dimensions
        if w.layers.len() != w.weights.len() + 1 {
            return Err(format!(
                "layers count ({}) != weights count + 1 ({})",
                w.layers.len(),
                w.weights.len() + 1
            ));
        }
        
        for i in 0..w.weights.len() {
            let expected = w.layers[i] * w.layers[i + 1];
            if w.weights[i].len() != expected {
                return Err(format!(
                    "Layer {}: expected {} weights ({}*{}), got {}",
                    i, expected, w.layers[i], w.layers[i + 1], w.weights[i].len()
                ));
            }
            if w.biases[i].len() != w.layers[i + 1] {
                return Err(format!(
                    "Layer {}: expected {} biases, got {}",
                    i, w.layers[i + 1], w.biases[i].len()
                ));
            }
        }
        
        Ok(Model { weights: w })
    }
    
    /// Forward pass — returns output vector.
    pub fn predict(&self, input: &[f64]) -> Result<Vec<f64>, String> {
        if input.len() != self.weights.layers[0] {
            return Err(format!(
                "Input size {} != expected {}",
                input.len(),
                self.weights.layers[0]
            ));
        }
        
        let mut current: Vec<f64> = input.to_vec();
        
        for i in 0..self.weights.weights.len() {
            let out_size = self.weights.layers[i + 1];
            let in_size = self.weights.layers[i];
            
            // Matrix multiply: output[j] = sum_k(input[k] * weight[j*in_size + k]) + bias[j]
            let mut next = vec![0.0; out_size];
            
            for j in 0..out_size {
                let mut sum = self.weights.biases[i][j];
                for k in 0..in_size {
                    sum += current[k] * self.weights.weights[i][j * in_size + k];
                }
                next[j] = sum;
            }
            
            // Apply activation
            current = self.weights.activations[i].forward(&next);
        }
        
        Ok(current)
    }
    
    /// Predict and return the class index with highest probability.
    pub fn predict_class(&self, input: &[f64]) -> Result<(usize, f64), String> {
        let output = self.predict(input)?;
        let (idx, &val) = output.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));
        Ok((idx, val))
    }
    
    pub fn input_size(&self) -> usize { self.weights.layers[0] }
    pub fn output_size(&self) -> usize { *self.weights.layers.last().unwrap_or(&0) }
    pub fn labels(&self) -> &[String] { &self.weights.labels }
}
