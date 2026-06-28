use std::f64;

/// Activation functions for neural network layers.
#[derive(Debug, Clone, serde::Deserialize)]
pub enum Activation {
    #[serde(rename = "relu")]
    ReLU,
    #[serde(rename = "sigmoid")]
    Sigmoid,
    #[serde(rename = "softmax")]
    Softmax,
    #[serde(rename = "linear")]
    Linear,
}

impl Activation {
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        match self {
            Activation::ReLU => x.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect(),
            Activation::Sigmoid => x.iter()
                .map(|&v| 1.0 / (1.0 + (-v.clamp(-20.0, 20.0)).exp()))
                .collect(),
            Activation::Softmax => {
                let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = x.iter().map(|&v| (v - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                exps.iter().map(|&v| v / sum).collect()
            }
            Activation::Linear => x.to_vec(),
        }
    }
}
