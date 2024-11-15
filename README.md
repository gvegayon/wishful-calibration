

# Wishful thinking in Calibration

- Calibration in ABMs (and simulation models in general) can be tricky.
- A common misconception is that the parameters of the model should
  match closely parameters in real life.
- In this document, I illustrate how that’s not the case and having
  “biased” calibration is expected.

## A simple model

To illustrate this, let’s think of a simple probit model:

$$
\Pr(y_i = 1) = \Phi^{-1}\left(\alpha + \mathbf{x}_i\theta\right)
$$

Where $\alpha = 1$, $\theta \equiv [2, 1]$, and $\mathbf{x}_i$ is a
vector of two components: one observed and one latent. The following
code generates the data and fits the model:

``` r
set.seed(2121)
n <- 1000
x <- cbind(
  x_observed = rnorm(n, 2),
  x_latent = rnorm(n, 3)
  )
theta <- cbind(c(2, -1))
y <- ((pnorm(1 + x %*% theta)) > runif(n)) |>
    as.numeric()

# Taking a look at the distribution
table(y)
```

    y
      0   1 
    182 818 

``` r
# Fitting the model (should be able to recover params)
glm(y ~ x, family=binomial(link="probit"))
```

    Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred


    Call:  glm(formula = y ~ x, family = binomial(link = "probit"))

    Coefficients:
    (Intercept)  xx_observed    xx_latent  
         1.2676       1.8844      -0.9792  

    Degrees of Freedom: 999 Total (i.e. Null);  997 Residual
    Null Deviance:      948.8 
    Residual Deviance: 388.1    AIC: 394.1

Imagine we are building an ABM that tries to forecast $y$. We have
access to the observed data, but not to the latent data. We can only
calibrate the model using the observed data. The following code splits
the data into observed and target data:

``` r
n_obs <- ceiling(n * .5)
x_obs <- x[1:n_obs,]
y_obs <- y[1:n_obs]

x_target <- x[-c(1:n_obs), ]
y_target <- y[-c(1:n_obs)]
```

The following code defines the MLE estimation function:

``` r
mle <- function(fn, par) {
  optim(
    par = par, 
    fn = fn,
    method = "BFGS",
    control = list(fnscale = -1),
    x_obs = x_obs,
    y_obs = y_obs 
  )
}
```

## Attempt one: Ideal world (we know all)

In an ideal world we get to know the data-generating process fully.
Here, the Maximum Likelihood estimation \[MLE\] should be able to
recover the parameters of the model **and** the Mean Absolute Error
\[MAE\] should be the lowest possible.

``` r
# Log-likelihood function
likelihood_knows_all <- function(param, y_obs, x_obs) {
  y_pred <- pnorm(
    param[1] + param[2] * x_obs[,1] + param[3] * x_obs[,2]
  )
  
  ifelse(y_obs == 1, y_pred, 1 - y_pred) |>
    log() |>
    sum()
}

# MLE estimation
(res_ideal <- mle(par = c(0, 0, 0), fn = likelihood_knows_all))
```

    $par
    [1]  1.477145  1.885586 -1.034879

    $value
    [1] -92.14649

    $counts
    function gradient 
          42       12 

    $convergence
    [1] 0

    $message
    NULL

``` r
# Computing the mae (mean absolute error)
y_pred <- pnorm(
  res_ideal$par[1] +
  res_ideal$par[2] * x_target[,1] +
  res_ideal$par[3] * x_target[,2]
  )

(mae_ideal <- mean(abs(y_target - y_pred)))
```

    [1] 0.1263854

## Attempt two: Calibrating only using `x_observed`

Most of the time, we don’t know the full data-generating process. In
this case, we can only calibrate the model using the observed data. This
is the most common scenario in calibration. Particularly, we only know
`x_observed` and instead of using a probit model, we use a logistic
model (so it is simpler).

``` r
likelihood_knows_obs <- function(param, y_obs, x_obs) {
  y_pred <- plogis(
    param[1] + param[2] * x_obs[,1]
    )
  ifelse(y_obs == 1, y_pred, 1-y_pred) |>
    log() |>
    sum()
}
```

``` r
(res_observed <- mle(par = c(0, 0), fn = likelihood_knows_obs))
```

    $par
    [1] -2.207985  2.508618

    $value
    [1] -131.7429

    $counts
    function gradient 
          37       10 

    $convergence
    [1] 0

    $message
    NULL

``` r
# Computing the mae (mean absolute error)
y_pred <- plogis(
  res_observed$par[1] +
  res_observed$par[2] * x_target[,1]
  )

(mae_observed <- mean(abs(y_target - y_pred)))
```

    [1] 0.1696311

## Attempt three: Wishful calibration (moving x towards 2)

Finally, we can try to calibrate the model while fixing one of the model
parameters. Imagine that we only care about the parameter associated
with the observed data (`x_observed`) and we treat the intercept as a
free parameter; therefore, instead of estimating the parameter
associated with `x_observed`, we fix it to 2. Nonetheless, we will still
add a normal prior to the intercept that is centered around 1.

``` r
# Log-likelihood function
likelihood_wishful <- function(param, y_obs, x_obs) {
  y_pred <- plogis(
    param[1] + 2 * x_obs[,1]
    )
  ans <- ifelse(y_obs == 1, y_pred, 1-y_pred) |>
    log() |>
    sum()

  ans + dnorm(param[1], 1, 1, log = TRUE)
}

# MLE estimation
(res_wishful <- mle(par = c(0), fn = likelihood_wishful))
```

    $par
    [1] -1.490123

    $value
    [1] -137.8456

    $counts
    function gradient 
          18        6 

    $convergence
    [1] 0

    $message
    NULL

``` r
# Computing the mae (mean absolute error)
y_pred <- plogis(
  res_wishful$par[1] +
  2 * x_target[,1]
  )

(mae_wishful <- mean(abs(y_target - y_pred)))
```

    [1] 0.1820921

## Fully wishful thinking

In this final model, instead of doing any estimation, we will use the
known model parameters:

``` r
y_pred <- plogis(1 + theta[2] * x_target[,1])
(mae_fully_wishful <- mean(abs(y_target - y_pred)))
```

    [1] 0.7136377

## Comparing all results

| Model         | Intercept (bias) | Bias (x observed) | Bias (x latent) |    MAE |
|:--------------|-----------------:|------------------:|----------------:|-------:|
| Real          |           0.4771 |           -0.1144 |         -0.0349 | 0.1264 |
| Approx        |          -3.2080 |            0.5086 |              NA | 0.1696 |
| Wishful       |          -2.4901 |            0.0000 |              NA | 0.1821 |
| Fully Wishful |           0.0000 |            0.0000 |              NA | 0.7136 |

From this table:

- As expected, the “Real” model is the best performing and has the
  smallest bias of the three models.
- The “Wishful” model has a smaller bias in the intercept (because of
  the prior), but has a higher MAE than the two other models.
- The “Approx” model has a higher bias in the intercept, but it has
  better performance in prediction compared to the “Wishful” model.
- The “Fully Wishful” model is the model with 0 bias in intercept and
  `x_observed`, but it has the highest MAE.
