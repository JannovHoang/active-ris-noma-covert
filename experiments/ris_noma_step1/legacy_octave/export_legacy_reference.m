function export_legacy_reference(output_file)
  % Run from the Python project root: addpath("legacy_octave"); export_legacy_reference("octave_reference.mat");
  % Same four channel calls and seed convention as the original source; no PSO.
  if nargin < 1
    output_file = "octave_reference.mat";
  endif
  rand("seed", 7); randn("seed", 7);
  M=2000; N=32; Krician=1; kappa=0.1;
  har=rician_matrix(M,N,Krician);
  hrb=rician_matrix(M,N,Krician);
  hrc=rician_matrix(M,N,Krician);
  hrw=rician_matrix(M,N,Krician);
  phase=exp(1i .* (-angle(har)-angle(hrb)));
  X=[abs(sum(har.*hrb.*phase,2)).^2, abs(sum(har.*hrc.*phase,2)).^2, abs(sum(har.*hrw.*phase,2)).^2];
  Y=[abs(sum(hrb.*phase,2)).^2, abs(sum(hrc.*phase,2)).^2, abs(sum(hrw.*phase,2)).^2];
  PL_ar=norm([0,0]-[95,5])^2.2;
  PL_rb=norm([95,5]-[110,0])^2.3;
  PL_rc=norm([95,5]-[112,2])^2.3;
  PL_rw=norm([95,5]-[100,0])^2.3;
  % Original units: mW. Python comparison converts every power to W.
  sigma_r2=10^(-70/10); sigma_b2=sigma_r2; sigma_c2=sigma_r2; sigma_w2=sigma_r2;
  B=10^(40/10)-N*(10^(-10/10)+10^(-5/10));
  candidates=[0.10,0.50; 0.20,0.30; 0.05,0.95; 0.49,0.05];
  [alpha3,beta3]=gamma_params(X(:,3)); [alpha4,beta4]=gamma_params(Y(:,3));
  expected=zeros(rows(candidates),5);
  for j=1:rows(candidates)
    mb=candidates(j,1); s=candidates(j,2);
    Pa=(1-s)*B; PRIS=s*B;
    rho2=PRIS/(N*(Pa/PL_ar+sigma_r2));
    [Rb,Rc]=compute_rates_mc(mb*Pa,(1-mb)*Pa,Pa,rho2,kappa, ...
      X(:,1),Y(:,1),X(:,2),Y(:,2),PL_ar,PL_rb,PL_rc,sigma_r2,sigma_b2,sigma_c2);
    [DEP,lambda_mw]=compute_dep_min_moment_matching((1-mb)*Pa,Pa,rho2,kappa, ...
      alpha3,beta3,alpha4,beta4,PL_ar,PL_rw,sigma_r2,sigma_w2,300);
    expected(j,:)=[rho2,Rb,Rc,DEP,lambda_mw];
  endfor
  schema_version=1;
  save("-mat7-binary",output_file,"schema_version","har","hrb","hrc","hrw","phase","X","Y","candidates","expected");
  fprintf("Saved common channel bank and original-formula outputs: %s\n",output_file);
  fprintf("Expected columns: rho2, Rb, Rc_Carol_only, DEP_grid, lambda_mW\n");
endfunction

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