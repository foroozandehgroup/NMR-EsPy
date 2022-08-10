function [magnet, sfo] = get_magnet_and_sfo(field, field_unit, channel)
    gamma = spin(channel);
    if field_unit == "MHz"
        magnet = (2e6 * pi * field) / gamma;
        sfo = field;
    else
        magnet = field;
        sfo = (field * gamma) / (2e6 * pi);
    end
end
