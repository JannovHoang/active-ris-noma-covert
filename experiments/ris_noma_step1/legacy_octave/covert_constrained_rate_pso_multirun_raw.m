function covert_constrained_rate_pso_multirun_raw()
  clc;
  clear;
  close all;
  more off;


  % COVERT-CONSTRAINED RATE MAXIMIZATION FOR ACTIVE RIS-NOMA

  % 0. MULTI-RUN SETTINGS

  % Keep the same channel samples for all PSO runs.

  fixed_channel_seed = 7;

  % Different PSO seeds, change to 1:5 for quick test.
  seed_list = 1:10;
  num_runs = length(seed_list);


  % 1. SYSTEM PARAMETERS

  N = 32;
  Krician = 1;
  kappa = 0.1;


  optimize_Ptot = false;
  fixed_Ptot_dBm = 40;
  Ptot_min_dBm = 30;
  Ptot_max_dBm = 44;

  % Noise power, dBm
  sigma_r2_dBm = -70;
  sigma_b2_dBm = -70;
  sigma_c2_dBm = -70;
  sigma_w2_dBm = -70;

  sigma_r2 = dbm2mw(sigma_r2_dBm);
  sigma_b2 = dbm2mw(sigma_b2_dBm);
  sigma_c2 = dbm2mw(sigma_c2_dBm);
  sigma_w2 = dbm2mw(sigma_w2_dBm);

  % Active RIS circuit power
  PS_dBm = -10;
  PDC_dBm = -5;
  PS = dbm2mw(PS_dBm);
  PDC = dbm2mw(PDC_dBm);

  % Path-loss exponents
  eps_ar = 2.2;
  eps_rb = 2.3;
  eps_rc = 2.3;
  eps_rw = 2.3;


  % Node coordinates
 % Format: [x, y] in meters.

  Alice  = [0, 0];
  Bob    = [110, 0];
  Carol  = [112, 2];
  Willie = [100, 0];

  xRIS = 95;
  yRIS = 5;
  RIS = [xRIS, yRIS];

  d_ar = norm(Alice - RIS);
  d_rb = norm(RIS - Bob);
  d_rc = norm(RIS - Carol);
  d_rw = norm(RIS - Willie);

  PL_ar = d_ar^eps_ar;
  PL_rb = d_rb^eps_rb;
  PL_rc = d_rc^eps_rc;
  PL_rw = d_rw^eps_rw;

  % QoS constraint for Carol
  Rc_min = 0.20;  % bps/Hz

  % Covertness constraints.
   xi_list = [0.75, 0.80, 0.85, 0.90, 0.95];

  % Monte Carlo samples and lambda grid
  Nmc = 2000;
  nLambda = 300;


  % 2. GENERATE CHANNELS ONCE
  % The channel is fixed for all PSO seeds.
  % average reflect PSO stability, not random changes in the channel.
  rand("seed", fixed_channel_seed);
  randn("seed", fixed_channel_seed);

  [Xb0, Yb0, Xc0, Yc0, Xw0, Yw0] = generate_channel_gains(Nmc, N, Krician);
  [alpha3, beta3_base] = gamma_params(Xw0);
  [alpha4, beta4_base] = gamma_params(Yw0);


  % 3. PSO SETTINGS

  swarm_size = 30;
  max_iter = 60;

  % PSO coefficients
  inertia_max = 0.90;
  inertia_min = 0.40;
  c1 = 1.50;
  c2 = 1.50;

  % Optional constriction factor.
  % use_chi = false means  Velocity is controlled by decreasing inertia and velocity clamping.
  use_chi = false;
  chi = 0.729;


  if optimize_Ptot
    dim = 3;
    lb = [0.05, 0.05, Ptot_min_dBm];
    ub = [0.49, 0.95, Ptot_max_dBm];
  else
    dim = 2;
    lb = [0.05, 0.05];
    ub = [0.49, 0.95];
  endif

  fprintf("============================================================\n");
  fprintf(" MULTI-RUN COVERT-CONSTRAINED BOB RATE MAXIMIZATION USING PSO\n");
  fprintf("============================================================\n");
  fprintf("Objective: maximize Rb\n");
  fprintf("Constraints: DEP_min >= xi, Rc >= Rc_min, mb < mc\n");
  fprintf("N = %d, kappa = %.4f, RIS = (%.2f, %.2f) m\n", N, kappa, xRIS, yRIS);
  fprintf("Rc_min = %.4f bps/Hz\n", Rc_min);
  fprintf("Nmc = %d, nLambda = %d\n", Nmc, nLambda);
  fprintf("PSO swarm size = %d, max_iter = %d\n", swarm_size, max_iter);
  fprintf("Fixed channel seed = %d\n", fixed_channel_seed);
  fprintf("Number of PSO runs = %d\n", num_runs);
  fprintf("PSO seeds: ");
  fprintf("%d ", seed_list);
  fprintf("\n");
  fprintf("Raw PSO variables are used, no [0,1] normalization.\n");

  if optimize_Ptot
    fprintf("Variables: mb, s, Ptot_dBm\n");
    fprintf("Ptot range = %.2f to %.2f dBm\n", Ptot_min_dBm, Ptot_max_dBm);
  else
    fprintf("Variables: mb, s\n");
    fprintf("Ptot fixed = %.2f dBm\n", fixed_Ptot_dBm);
  endif

  if use_chi
    fprintf("Constriction factor chi = %.3f is enabled.\n", chi);
  else
    fprintf("Constriction factor chi is disabled. Using decreasing inertia + velocity clamping.\n");
  endif

  fprintf("============================================================\n\n");

  % Result table per run:
  % 1 xi, 2 feasible, 3 best_Rb, 4 Rc, 5 DEP, 6 mb, 7 mc, 8 s, 9 Ptot_dBm, 10 rho, 11 fitness
  ALL_RESULTS = NaN(num_runs, length(xi_list), 11);
  ALL_CONV = NaN(num_runs, length(xi_list), max_iter);


  % 4. RUN MULTIPLE PSO RUNS

  for run_idx = 1:num_runs
    seed_id = seed_list(run_idx);

    rand("seed", seed_id);
    randn("seed", seed_id);

    fprintf("\n============================================================\n");
    fprintf(" PSO RUN %d/%d, seed = %d\n", run_idx, num_runs, seed_id);
    fprintf("============================================================\n");

    RESULT_RUN = NaN(length(xi_list), 11);
    CONV_RUN = NaN(length(xi_list), max_iter);


    % Run PSO for each covertness level xi

    for q = 1:length(xi_list)
      xi = xi_list(q);

      % Initialize swarm
      X = zeros(swarm_size, dim);
      V = zeros(swarm_size, dim);

      for i = 1:swarm_size
        X(i,:) = lb + rand(1, dim) .* (ub - lb);
        V(i,:) = 0.10 * (ub - lb) .* (2*rand(1, dim)-1);
      endfor

      pbest_X = X;
      pbest_fit = -Inf(swarm_size, 1);
      pbest_info = NaN(swarm_size, 8); % Rb, Rc, DEP, mb, mc, s, Ptot, rho

      gbest_X = X(1,:);
      gbest_fit = -Inf;
      gbest_info = NaN(1, 8);

      for iter = 1:max_iter
        inertia = inertia_max - (inertia_max - inertia_min) * (iter - 1) / max(1, max_iter - 1);

        for i = 1:swarm_size
          [fit, info] = evaluate_candidate( ...
            X(i,:), optimize_Ptot, fixed_Ptot_dBm, xi, Rc_min, ...
            N, PS, PDC, kappa, ...
            Xb0, Yb0, Xc0, Yc0, Xw0, Yw0, ...
            alpha3, beta3_base, alpha4, beta4_base, ...
            PL_ar, PL_rb, PL_rc, PL_rw, ...
            sigma_r2, sigma_b2, sigma_c2, sigma_w2, nLambda);

          if fit > pbest_fit(i)
            pbest_fit(i) = fit;
            pbest_X(i,:) = X(i,:);
            pbest_info(i,:) = info;
          endif

          if fit > gbest_fit
            gbest_fit = fit;
            gbest_X = X(i,:);
            gbest_info = info;
          endif
        endfor

        CONV_RUN(q, iter) = gbest_fit;

        % Update velocities and positions
        for i = 1:swarm_size
          r1 = rand(1, dim);
          r2 = rand(1, dim);

          newV = inertia * V(i,:) ...
               + c1 * r1 .* (pbest_X(i,:) - X(i,:)) ...
               + c2 * r2 .* (gbest_X - X(i,:));

          if use_chi
            newV = chi * newV;
          endif

          V(i,:) = newV;

          % Limit velocity to avoid too large jumps
          vmax = 0.25 * (ub - lb);
          V(i,:) = max(min(V(i,:), vmax), -vmax);

          X(i,:) = X(i,:) + V(i,:);
          X(i,:) = max(min(X(i,:), ub), lb);
        endfor
      endfor

      % Extract best info
      Rb_best = gbest_info(1);
      Rc_best = gbest_info(2);
      DEP_best = gbest_info(3);
      mb_best = gbest_info(4);
      mc_best = gbest_info(5);
      s_best = gbest_info(6);
      Ptot_best = gbest_info(7);
      rho_best = gbest_info(8);

      feasible = (DEP_best >= xi) && (Rc_best >= Rc_min) && (mb_best < mc_best);

      RESULT_RUN(q,:) = [xi, feasible, Rb_best, Rc_best, DEP_best, mb_best, mc_best, s_best, Ptot_best, rho_best, gbest_fit];

      fprintf("seed = %d | xi = %.2f | feasible = %d | Rb* = %.6f | Rc = %.6f | DEP = %.6f | mb = %.4f | mc = %.4f | s = %.4f | Ptot = %.2f dBm\n", ...
        seed_id, xi, feasible, Rb_best, Rc_best, DEP_best, mb_best, mc_best, s_best, Ptot_best);
    endfor

    % Save this run into aggregate arrays
    for q = 1:length(xi_list)
      ALL_RESULTS(run_idx, q, :) = RESULT_RUN(q, :);
      ALL_CONV(run_idx, q, :) = CONV_RUN(q, :);
    endfor

    % Save per-run data to avoid losing results
    run_mat_name = sprintf("covert_constrained_pso_raw_seed%d.mat", seed_id);
    save("-mat", run_mat_name, ...
         "RESULT_RUN", "CONV_RUN", "xi_list", "Rc_min", ...
         "fixed_Ptot_dBm", "optimize_Ptot", ...
         "swarm_size", "max_iter", "seed_id", "fixed_channel_seed");
    fprintf("Saved run data: %s\n", run_mat_name);
  endfor


  % 5. MULTI-RUN SUMMARY

  MEAN_RESULTS = squeeze(mean(ALL_RESULTS, 1));
  STD_RESULTS  = squeeze(std(ALL_RESULTS, 0, 1));

  feasible_counts = zeros(length(xi_list), 1);
  for q = 1:length(xi_list)
    feasible_counts(q) = sum(ALL_RESULTS(:, q, 2) == 1);
  endfor

  fprintf("\n============================================================\n");
  fprintf(" MULTI-RUN SUMMARY TABLE: MEAN +/- STD OVER %d PSO RUNS\n", num_runs);
  fprintf("============================================================\n");
  fprintf(" xi    FeasCount  Rb_mean  Rb_std   Rc_mean  DEP_mean  mb_mean  mc_mean  s_mean  s_std   Ptot_mean\n");

  for q = 1:length(xi_list)
    fprintf(" %.2f   %2d/%-2d    %.6f  %.6f  %.6f  %.6f  %.4f   %.4f   %.4f  %.4f  %.2f\n", ...
      xi_list(q), feasible_counts(q), num_runs, ...
      MEAN_RESULTS(q,3), STD_RESULTS(q,3), ...
      MEAN_RESULTS(q,4), MEAN_RESULTS(q,5), ...
      MEAN_RESULTS(q,6), MEAN_RESULTS(q,7), ...
      MEAN_RESULTS(q,8), STD_RESULTS(q,8), ...
      MEAN_RESULTS(q,9));
  endfor

  fprintf("============================================================\n\n");

  % Save aggregated data
  save("-mat", "covert_constrained_pso_raw_multirun_results.mat", ...
       "ALL_RESULTS", "ALL_CONV", "MEAN_RESULTS", "STD_RESULTS", ...
       "feasible_counts", "xi_list", "Rc_min", ...
       "fixed_Ptot_dBm", "optimize_Ptot", ...
       "swarm_size", "max_iter", "seed_list", "fixed_channel_seed", ...
       "lb", "ub", "use_chi", "chi");


  % 6. AVERAGED PLOTS

  feasible_idx = MEAN_RESULTS(:,2) >= 0.5;

  figure;
  plot(MEAN_RESULTS(feasible_idx,1), MEAN_RESULTS(feasible_idx,3), "-o", "linewidth", 1.5);
  xlabel("Covertness threshold \xi");
  ylabel("Mean maximum Bob rate R_b^* (bps/Hz)");
  title(sprintf("Mean covert-constrained Bob rate over %d PSO runs", num_runs));
  grid on;
  print("-dpng", "fig_pso_raw_Rb_vs_xi_mean.png");

  figure;
  plot(MEAN_RESULTS(feasible_idx,1), MEAN_RESULTS(feasible_idx,5), "-s", "linewidth", 1.5);
  hold on;
  plot(MEAN_RESULTS(feasible_idx,1), MEAN_RESULTS(feasible_idx,1), "k--", "linewidth", 1.2);
  hold off;
  xlabel("Covertness threshold \xi");
  ylabel("Mean achieved DEP_{min}");
  title(sprintf("Mean achieved covertness over %d PSO runs", num_runs));
  legend("Mean achieved DEP_{min}", "Required \xi", "location", "northeast");
  grid on;
  print("-dpng", "fig_pso_raw_DEP_vs_xi_mean.png");

  figure;
  plot(MEAN_RESULTS(feasible_idx,1), MEAN_RESULTS(feasible_idx,6), "-o", "linewidth", 1.5);
  hold on;
  plot(MEAN_RESULTS(feasible_idx,1), MEAN_RESULTS(feasible_idx,8), "-s", "linewidth", 1.5);
  hold off;
  xlabel("Covertness threshold \xi");
  ylabel("Mean optimized parameter");
  title(sprintf("Mean optimized mb and s over %d PSO runs", num_runs));
  legend("mean m_b^*", "mean s^*", "location", "northeast");
  grid on;
  print("-dpng", "fig_pso_raw_params_vs_xi_mean.png");

  % Average convergence of the middle xi value, usually xi = 0.85
  mid = ceil(length(xi_list)/2);
  mean_conv = squeeze(mean(ALL_CONV(:, mid, :), 1))';
  std_conv  = squeeze(std(ALL_CONV(:, mid, :), 0, 1))';

  iter_vec = 1:max_iter;

  figure;
  plot(iter_vec, mean_conv, "linewidth", 1.8);
  hold on;
  plot(iter_vec, mean_conv + std_conv, "k--", "linewidth", 1.0);
  plot(iter_vec, mean_conv - std_conv, "k--", "linewidth", 1.0);
  hold off;
  xlabel("Iteration");
  ylabel("Average best PSO fitness");
  title(sprintf("Average PSO convergence for xi = %.2f over %d runs", xi_list(mid), num_runs));
  legend("Mean best fitness", "Mean + std", "Mean - std", "location", "southeast");
  grid on;
  print("-dpng", "fig_pso_raw_convergence_average_xi085.png");

  fprintf("Saved aggregated data: covert_constrained_pso_raw_multirun_results.mat\n");
  fprintf("Saved averaged figures:\n");
  fprintf("  fig_pso_raw_Rb_vs_xi_mean.png\n");
  fprintf("  fig_pso_raw_DEP_vs_xi_mean.png\n");
  fprintf("  fig_pso_raw_params_vs_xi_mean.png\n");
  fprintf("  fig_pso_raw_convergence_average_xi085.png\n");
  fprintf("\nDONE.\n");

endfunction



% Evaluate one PSO candidate.
% Fitness is designed for constrained maximization:
%   feasible candidates: fitness = Rb
%   infeasible candidates: fitness = Rb - large_penalty

function [fitness, info] = evaluate_candidate( ...
    x, optimize_Ptot, fixed_Ptot_dBm, xi, Rc_min, ...
    N, PS, PDC, kappa, ...
    Xb0, Yb0, Xc0, Yc0, Xw0, Yw0, ...
    alpha3, beta3_base, alpha4, beta4_base, ...
    PL_ar, PL_rb, PL_rc, PL_rw, ...
    sigma_r2, sigma_b2, sigma_c2, sigma_w2, nLambda)

  mb = x(1);
  s  = x(2);
  mc = 1 - mb;

  if optimize_Ptot
    Ptot_dBm = x(3);
  else
    Ptot_dBm = fixed_Ptot_dBm;
  endif

  Ptot = dbm2mw(Ptot_dBm);
  B = Ptot - N * (PS + PDC);

  % Default bad values
  Rb = -Inf;
  Rc = -Inf;
  DEP_min = -Inf;
  rho = NaN;

  if B <= 0 || mb <= 0 || mc <= 0 || mb >= mc || s <= 0 || s >= 1
    violation = 10;
    fitness = -1e6 - 1e4 * violation;
    info = [Rb, Rc, DEP_min, mb, mc, s, Ptot_dBm, rho];
    return;
  endif

  Pa_act   = (1 - s) * B;
  PRIS_act = s * B;

  if Pa_act <= 0 || PRIS_act <= 0
    violation = 10;
    fitness = -1e6 - 1e4 * violation;
    info = [Rb, Rc, DEP_min, mb, mc, s, Ptot_dBm, rho];
    return;
  endif

  rho2 = PRIS_act / (N * (Pa_act / PL_ar + sigma_r2));
  rho = sqrt(rho2);

  if rho2 <= 0 || !isfinite(rho2)
    violation = 10;
    fitness = -1e6 - 1e4 * violation;
    info = [Rb, Rc, DEP_min, mb, mc, s, Ptot_dBm, rho];
    return;
  endif

  Pb = mb * Pa_act;
  Pc = mc * Pa_act;

  [Rb, Rc] = compute_rates_mc( ...
    Pb, Pc, Pa_act, rho2, kappa, ...
    Xb0, Yb0, Xc0, Yc0, ...
    PL_ar, PL_rb, PL_rc, ...
    sigma_r2, sigma_b2, sigma_c2);

  [DEP_min, lambda_star] = compute_dep_min_moment_matching( ...
    Pc, Pa_act, rho2, kappa, ...
    alpha3, beta3_base, alpha4, beta4_base, ...
    PL_ar, PL_rw, sigma_r2, sigma_w2, nLambda);

  % Constraint violations. All should be <= 0 for feasibility.
  v_dep = max(0, xi - DEP_min);
  v_qos = max(0, Rc_min - Rc);
  v_noma = max(0, mb - mc + 1e-6);

  total_violation = v_dep + v_qos + v_noma;

  % Strong penalty: infeasible solutions should not beat feasible ones.
  penalty_scale = 1000;
  fitness = Rb - penalty_scale * total_violation;

  info = [Rb, Rc, DEP_min, mb, mc, s, Ptot_dBm, rho];

endfunction


% Helper: dBm to mW

function y = dbm2mw(x_dBm)
  y = 10.^(x_dBm ./ 10);
endfunction


% Helper: mW to dBm

function y = mw2dbm(x_mW)
  x_mW = max(x_mW, realmin);
  y = 10 .* log10(x_mW);
endfunction


% Generate Rician channels and effective gains

function [Xb0, Yb0, Xc0, Yc0, Xw0, Yw0] = generate_channel_gains(Nmc, N, Krician)

  har = rician_matrix(Nmc, N, Krician);
  hrb = rician_matrix(Nmc, N, Krician);
  hrc = rician_matrix(Nmc, N, Krician);
  hrw = rician_matrix(Nmc, N, Krician);

  % RIS phase aligned to Bob
  theta = -angle(har) - angle(hrb);
  v = exp(1i .* theta);

  Hab0 = sum(har .* hrb .* v, 2);
  Hac0 = sum(har .* hrc .* v, 2);
  Haw0 = sum(har .* hrw .* v, 2);

  Gb0 = sum(hrb .* v, 2);
  Gc0 = sum(hrc .* v, 2);
  Gw0 = sum(hrw .* v, 2);

  Xb0 = abs(Hab0).^2;
  Xc0 = abs(Hac0).^2;
  Xw0 = abs(Haw0).^2;

  Yb0 = abs(Gb0).^2;
  Yc0 = abs(Gc0).^2;
  Yw0 = abs(Gw0).^2;

endfunction


% Rician fading generator

function h = rician_matrix(M, N, Krician)

  los_phase = 2 * pi * rand(1, N);
  LOS = repmat(exp(1i .* los_phase), M, 1);
  NLOS = (randn(M, N) + 1i .* randn(M, N)) ./ sqrt(2);

  h = sqrt(Krician / (Krician + 1)) .* LOS ...
      + sqrt(1 / (Krician + 1)) .* NLOS;

endfunction


% Gamma moment matching

function [alpha, beta] = gamma_params(X)

  m = mean(X);
  v = var(X, 1);

  m = max(m, realmin);
  v = max(v, realmin);

  alpha = m^2 / v;
  beta = v / m;

endfunction


% Bob and Carol rates

function [Rb, Rc] = compute_rates_mc( ...
    Pb, Pc, Pa, rho2, kappa, ...
    Xb0, Yb0, Xc0, Yc0, ...
    PL_ar, PL_rb, PL_rc, ...
    sigma_r2, sigma_b2, sigma_c2)

  Xb = rho2 .* Xb0;
  Yb = rho2 .* Yb0;
  Xc = rho2 .* Xc0;
  Yc = rho2 .* Yc0;

  gamma_b_num = (Pb ./ (PL_ar * PL_rb)) .* Xb;
  gamma_b_den = (kappa^2 * Pa ./ (PL_ar * PL_rb)) .* Xb ...
                + (sigma_r2 ./ PL_rb) .* Yb ...
                + sigma_b2;
  gamma_b = gamma_b_num ./ gamma_b_den;

  gamma_c_num = (Pc ./ (PL_ar * PL_rc)) .* Xc;
  gamma_c_den = ((kappa^2 * Pa + Pb) ./ (PL_ar * PL_rc)) .* Xc ...
                + (sigma_r2 ./ PL_rc) .* Yc ...
                + sigma_c2;
  gamma_c = gamma_c_num ./ gamma_c_den;

  Rb = mean(log2(1 + gamma_b));
  Rc = mean(log2(1 + gamma_c));

endfunction


% Minimum DEP using moment matching and lambda search

function [DEP_min, lambda_star] = compute_dep_min_moment_matching( ...
    Pc, Pa, rho2, kappa, ...
    alpha3, beta3_base, alpha4, beta4_base, ...
    PL_ar, PL_rw, sigma_r2, sigma_w2, nLambda)

  beta3 = rho2 * beta3_base;
  beta4 = rho2 * beta4_base;

  a1 = (1 + kappa^2) * Pc / (PL_ar * PL_rw);
  a2 = (1 + kappa^2) * Pa / (PL_ar * PL_rw);
  b  = sigma_r2 / PL_rw;

  mu1  = a1 * alpha3 * beta3 + b * alpha4 * beta4;
  var1 = a1^2 * alpha3 * beta3^2 + b^2 * alpha4 * beta4^2;

  mu2  = a2 * alpha3 * beta3 + b * alpha4 * beta4;
  var2 = a2^2 * alpha3 * beta3^2 + b^2 * alpha4 * beta4^2;

  mu1 = max(mu1, realmin);
  mu2 = max(mu2, realmin);
  var1 = max(var1, realmin);
  var2 = max(var2, realmin);

  delta1 = mu1^2 / var1;
  tau1   = var1 / mu1;

  delta2 = mu2^2 / var2;
  tau2   = var2 / mu2;

  xMax = max([mu1 + 8 * sqrt(var1), mu2 + 8 * sqrt(var2), realmin]);
  xGrid = linspace(0, xMax, nLambda);

  upper1 = gammainc(xGrid ./ tau1, delta1, "upper");
  upper2 = gammainc(xGrid ./ tau2, delta2, "upper");

  DEP = upper1 + 1 - upper2;

  [DEP_min, idx] = min(DEP);
  lambda_star = sigma_w2 + xGrid(idx);

endfunction