function [fid, sfo] = onedim_sim(shifts, couplings, pts, sw, offset, field, nucleus)
    sfo = get_sfo(field, nucleus, offset);
    sys.magnet = field;
    nspins = length(shifts);
    isotopes = cell(nspins, 1);
    for i = 1:nspins
        isotopes{i} = nucleus;
    end
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
    parameters.offset = offset;
    parameters.sweep = sw;
    parameters.npoints = pts;
    parameters.spins = {nucleus};
    parameters.coil=state(spin_system, 'L+', nucleus);
    parameters.rho0=state(spin_system, 'L+', nucleus);

    % Simulation
    fid = liquid(spin_system, @acquire, parameters, 'nmr');
end
