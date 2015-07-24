--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'punxnet'

return function(args)
    args.n_units        = {16, 24, 32}
    args.filter_size    = {5, 3, 2}
    args.filter_stride  = {4, 3, 2}
    args.n_hid          = {256, 256}
    args.nl             = nn.PReLU

    return create_network(args)
end

