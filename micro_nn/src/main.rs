use domain_classifier::model::Model;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: domain-classifier <model.json> [features...]");
        eprintln!("       domain-classifier <model.json> --test");
        eprintln!();
        eprintln!("Features: grid_w grid_h n_colors entropy spatial_var objects_ratio");
        eprintln!("Example: domain-classifier model.json 1.0 1.0 0.3 0.35 0.45 0.02");
        std::process::exit(1);
    }
    
    let model_path = &args[1];
    let json = fs::read_to_string(model_path)
        .unwrap_or_else(|e| {
            eprintln!("Failed to read {}: {}", model_path, e);
            std::process::exit(1);
        });
    
    let model = Model::from_json(&json).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {}", e);
        std::process::exit(1);
    });
    
    println!("Model: {:?} → {:?}", model.input_size(), model.output_size());
    println!("Labels: {:?}", model.labels());
    
    if args.len() >= 2 + model.input_size() {
        // Predict from command-line features
        let features: Vec<f64> = args[2..2 + model.input_size()]
            .iter()
            .map(|s| s.parse::<f64>().unwrap_or_else(|_| {
                eprintln!("Invalid feature: {}", s);
                std::process::exit(1);
            }))
            .collect();
        
        let output = model.predict(&features).unwrap();
        let (class, conf) = model.predict_class(&features).unwrap();
        
        println!("\nInput:   {:?}", features);
        println!("Output:  {:?}", output.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
        
        let label = model.labels().get(class).map(|s| s.as_str()).unwrap_or("?");
        println!("Class:   {} ({}) confidence={:.4}", class, label, conf);
    } else {
        // Test mode: run on known patterns
        println!("\nTesting known ARC-AGI-3 patterns:");
        
        let test_cases: Vec<(&str, &[f64])> = vec![
            ("LS20 (movement)",   &[1.0, 1.0, 0.3, 0.35, 0.45, 0.02]),
            ("VC33 (rotation)",   &[0.47, 0.47, 0.5, 0.55, 0.75, 0.06]),
            ("FT09 (transform)",  &[0.16, 0.16, 0.8, 0.75, 0.35, 0.20]),
            ("CD82 (hybrid)",     &[0.63, 0.63, 0.6, 0.55, 0.60, 0.08]),
        ];
        
        for (name, features) in &test_cases {
            let (class, conf) = model.predict_class(features).unwrap();
            let label = model.labels().get(class).map(|s| s.as_str()).unwrap_or("?");
            println!("  {:<20} → {} (conf={:.4})", name, label, conf);
        }
    }
}
