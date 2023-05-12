function [fid, sfo] = hsqc_sim(shifts, couplings, isotopes, field, pts, sw, off, nuclei)
    sfo = zeros(2, 1);
    for i = 1:2
        sfo(i, 1) = get_sfo(field, nuclei{i}, off(i));
    sys.magnet = field;
    nspins = length(shifts);
    sys.isotopes = isotopes;

    % Interations
    inter.zeeman.scalar = shifts;
    inter.coupling.scalar = get_couplings(nspins, couplings);

    % Basis set
    bas.formalism = 'sphten-liouv';
    bas.approximation = 'IK-2';
    bas.connectivity = 'scalar_couplings';
    bas.space_level = 1;

    % Spinach housekeeping
    spin_system = create(sys, inter);
    spin_system = basis(spin_system, bas);

    % Sequence parameters
    parameters.J = 120;
    parameters.sweep = sw;
    parameters.offset = off;
    parameters.npoints = pts;
    parameters.spins = nuclei;
    parameters.decouple_f1 = {nuclei{2}};
    parameters.decouple_f2 = {nuclei{1}};

    % Simulation
    fid = liquid(spin_system, @hsqc, parameters, 'nmr');
end
