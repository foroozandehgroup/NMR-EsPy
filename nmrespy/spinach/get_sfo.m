function sfo = get_sfo(field, channel)
    gamma = spin(channel);
    sfo = (field * gamma) / (2e6 * pi);
end
