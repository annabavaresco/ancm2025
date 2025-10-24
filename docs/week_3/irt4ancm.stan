data {
  int<lower=1> M;  // number of observations
  int<lower=1> N;  // number of participants
  int<lower=1> I;  // number of song segments
  int<lower=1> J;  // number of songs
  array[M] int<lower=0,upper=1> is_recognised;  // was the segment recognised?
  array[M] int<lower=0,upper=1> is_verified;    // was the segment verified correctly?
  array[M] int<lower=1,upper=N> participant;    // participant number
  array[M] int<lower=1,upper=I> seg;            // segment number
  array[M] int<lower=1,upper=I> song;           // segment number
  array[M] int<lower=0,upper=1> continuation_correctness;  // did the verification segment restart in the correct place?
  vector<lower=0>[M] recognition_time;            // how long did it take to recognise the segment?
  matrix[I,10] audio_features;                             // audio features for each segment
  vector[N] sophistication;                                // Goldsmith's music sophistication for each participant
}

// This is hopefully a Rasch model
parameters {
  real mu_delta;
  real<lower=0> sigma_theta;  // participant prior SD
  real<lower=0> sigma_delta;  // difficulty prior SD
  vector[N] theta;  // participant abilities
  vector[I] delta;  // segment difficulties

  // Adding discrimination (2PL)
  real<lower=0> sigma_alpha;  // difficulty prior SD
  vector[I] alpha;  // segment discrimination
  real mu_alpha;

}

model {
  // Hyperpriors
  mu_delta ~ std_normal();
  sigma_theta ~ std_normal(); // Stan automatically cuts off the negative values
  sigma_delta ~ std_normal(); // Stan automatically cuts off the negative values
  
    // 2PL
  mu_alpha ~ std_normal();
  sigma_alpha ~ std_normal(); // Stan automatically cuts off the negative values

  // Priors
  theta ~ normal(0, sigma_theta); // one ot the mus has to be zero
  delta ~ normal(mu_delta, sigma_delta);
  
  // 2PL
  alpha ~ normal(mu_alpha, sigma_alpha);

  // Data distribution
  for (m in 1:M) {
    // is_verified[m] ~ bernoulli_logit(theta[participant[m]] - delta[seg[m]]);
    is_verified[m] ~ bernoulli_logit(alpha[seg[m]]*(theta[participant[m]] - delta[seg[m]]));
  }
}
