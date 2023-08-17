function fid=jres_seq(spin_system,parameters,H,R,K)
    % Compose Liouvillian
    L = H + 1i * R + 1i * K; clear('H','R','K');
    % Coherent evolution timestep
    timestep1 = 1 / parameters.sweep(1);
    timestep2 = 1 / parameters.sweep(2);
    % Initial state
    rho = state(spin_system, 'Lz', parameters.spins{1});
    % Detection state
    coil = state(spin_system, 'L+', parameters.spins{1});
    % Get the pulse operator
    Lp = operator(spin_system, 'L+', parameters.spins{1});
    % Apply the first pulse
    rho = step(spin_system, (Lp + Lp') / 2, rho, pi / 2);
    % Run the F1 evolution
    rho = evolution(spin_system, L, [], rho, timestep1 / 2, parameters.npoints(1) - 1, 'trajectory');
    % Select "+1" coherence
    rho = coherence(spin_system, rho, {{parameters.spins{1}, -1}});
    % Apply the second pulse
    rho = step(spin_system, (Lp + Lp') / 2, rho, pi);
    % Select "-1" coherence
    rho = coherence(spin_system, rho, {{parameters.spins{1}, +1}});
    % Run the F1 evolution
    rho = evolution(spin_system, L, [], rho, timestep1 / 2, parameters.npoints(1) - 1, 'refocus');
    % Run the F2 evolution
    fid = evolution(spin_system, L, coil, rho, timestep2, parameters.npoints(2) - 1, 'observable');
end