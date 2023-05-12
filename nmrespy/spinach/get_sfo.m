function sfo = get_sfo(field, nucleus, offset)
    gamma = spin(nucleus);
    sfo = ((gamma * field / (2 * pi)) + offset) * 1e-6
end
