# Linear Regression in Rust

A simple linear regression implemented in Rust - because why not!  the meat of this
is the struct LinearRegression, representing a simple linear model.

```Rust
  struct LinearRegression {
      betas: Vec<f64>,
      bias: f64,
      x: Vec<Vec<f64>>,
      y: Vec<f64>
  }
```

Then we have several implemented methods, new, step, fit, and predict which
encapsulate the expected behavior.  step uses a probably incorrect version of
gradient descent.
