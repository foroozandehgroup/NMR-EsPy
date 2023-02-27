function fid=cpmg_seq(spin_system,parameters,H,R,K)
    % Compose Liouvillian
    L = H + 1i * R + 1i * K; clear('H','R','K');
    % Coherent evolution timestep
    timestep = 1 / parameters.sweep;
    % Initial state
    rho = state(spin_system, 'Lz', parameters.spins{1});
    % Detection state
    coil = state(spin_system, 'L-', parameters.spins{1});
    % Get the pulse operators
    Lp = operator(spin_system, 'L+', parameters.spins{1});
    Lx = (Lp + Lp') / 2;
    Ly = (Lp - Lp') / 2j;
    fid = zeros(parameters.increments, parameters.npoints, 'like', 1j);
    % Apply the first pulse
    rho = step(spin_system, Lx, rho, pi / 2);
    for n=1:parameters.increments
        % Evolve for tau
        rho = evolution(spin_system, L, [], rho, parameters.tau, 1, 'final');
        % % -1 coherence
        % rho = coherence(spin_system, rho, {{parameters.spins{1}, -1}});
        % y 180 pulse
        rho = step(spin_system, Ly, rho, pi);
        % Evolve for tau
        rho = evolution(spin_system, L, [], rho, parameters.tau, 1, 'final');
        % y 90 pulse
        rho = step(spin_system, Ly, rho, pi / 2);
        % Evolve for tau
        rho = evolution(spin_system, L, [], rho, parameters.tau, 1, 'final');
        % % +1 coherence
        % rho = coherence(spin_system, rho, {{parameters.spins{1}, +1}});
        % y 180 pulse
        rho = step(spin_system, Ly, rho, pi);
        % Evolve for tau
        rho = evolution(spin_system, L, [], rho, parameters.tau, 1, 'final');
        % % -1 coherence
        % rho = coherence(spin_system, rho, {{parameters.spins{1}, -1}});
        fid(n, :) = evolution(spin_system, L, coil, rho, timestep, parameters.npoints - 1, 'observable');
    end
end
