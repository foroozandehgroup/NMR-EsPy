function fid=sucrose_jres(np1,np2,homodecoupled);

    % Read the spin system properties (vacuum DFT calculation)
    options.min_j=1.0;
    [sys,inter]=g2spinach(gparse('sucrose.log'), {{'H','1H'}}, 31.8, options);

    if homodecoupled
        % To generate homodecoupled spin system,
        % simply set all scalar couplings to 0.
        inter.coupling.scalar = cell(22, 22);
        inter.coupling.scalar(:, :) = {0};
    end

    % Magnetic field
    BF1 = 300;  % 1H Larmor frequency (MHz)
    gamma_1H = 267.5222005 / (2 * pi);
    sys.magnet = BF1 / gamma_1H;  % B0 (Tesla)

    % Basis set
    bas.formalism = 'sphten-liouv';
    bas.approximation = 'IK-2';
    bas.connectivity = 'scalar_couplings';
    bas.space_level = 1;

    % Relaxation theory parameters
    inter.relaxation = {'redfield'};
    inter.equilibrium = 'zero';
    inter.rlx_keep = 'kite';
    inter.tau_c = {200e-12};

    % Sequence parameters
    parameters.offset = 1000;
    parameters.sweep = [40 2200];
    parameters.npoints = [np1 np2];
    parameters.spins = {'1H'};

    % Spinach housekeeping
    spin_system = create(sys, inter);
    spin_system = basis(spin_system, bas);

    % Simulation
    fid = liquid(spin_system, @jres_seq, parameters, 'nmr');

    if homodecoupled
        save(sprintf("sucrose_2dj_homo_%d_%d.mat", np1, np2), "fid");
    else
        save(sprintf("sucrose_2dj_%d_%d.mat", np1, np2), "fid");
    end

end
