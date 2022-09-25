function field = get_field(sfo, nucleus, offset)
    gamma = spin(nucleus);
    field = 2 * pi * (1e6 * sfo - offset) / gamma;
end
