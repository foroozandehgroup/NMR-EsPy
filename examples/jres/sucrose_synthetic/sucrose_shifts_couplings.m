% Read the spin system properties (vacuum DFT calculation)
function [shifts, couplings] = sucrose_shifts_couplings()
    options.min_j=1.0;
    [sys,inter]=g2spinach(gparse('sucrose.log'),...
                                     {{'H','1H'}},31.8,options);
    nspins = length(inter.zeeman.matrix);
    shifts = zeros(1, nspins);
    for i = 1:nspins
        m = inter.zeeman.matrix(i);
        shifts(i) = trace(m{1}) / 3;
    end
    couplings = [];
    for i = 1:nspins
        for j = 1:nspins
            if inter.coupling.scalar{i,j} ~= 0
                couplings = [couplings; [i j inter.coupling.scalar{i,j}]];
            end
        end
    end
end
