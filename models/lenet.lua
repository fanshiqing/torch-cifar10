require 'nn'

model = nn.Sequential()

model:add(nn.SpatialConvolution(3, 6, 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.SpatialConvolution(6, 16, 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.SpatialConvolution(16, 120, 5, 5))
model:add(nn.ReLU())

model:add(nn.View(120))
model:add(nn.Linear(120, 84))
model:add(nn.ReLU())
model:add(nn.Linear(84, 10))
model:add(nn.LogSoftMax())

return model
