function couplings = get_couplings(nspins, coups)
    couplings = cell(nspins, nspins);
    for elem = coups
        couplings{elem{1}{1}, elem{1}{2}} = elem{1}{3};
    end
