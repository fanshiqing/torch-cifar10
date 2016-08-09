require 'torch'
require 'paths'

cifar10 = {}
cifar10.path_remote = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
cifar10.path_dataset = paths.basename(cifar10.path_remote)
cifar10.path_folder = 'cifar-10-batches-t7'

function cifar10.download()
    local flag = false
    for i=0,4 do
        if not paths.filep(paths.concat(cifar10.path_folder, 'data_batch_' .. (i + 1) .. '.t7')) then
            flag = true
        end
    end

    if not paths.filep(paths.concat(cifar10.path_folder, 'test_batch.t7')) then
        flag = true
    end

    if floag then
        local remote = cifar10.path_remote
        local tar = paths.basename(remote)
        os.execute('wget ' .. remote)
        os.execute('tar xvf ' .. tar)
        os.execute('rm ' .. tar)
    end
end

function cifar10.loadTrainSet(maxLoad, geometry)
    cifar10.download()

    local trainData = {
        data = torch.Tensor(50000, 3072),
        labels = torch.Tensor(50000)
    }

    for i=0,4 do
        local temp = torch.load(paths.concat(cifar10.path_folder, 'data_batch_' .. (i + 1) .. '.t7'), 'ascii')
        trainData.data[{ {i * 10000 + 1, (i + 1) * 10000} }] = temp.data:t()
        trainData.labels[{ {i * 10000 + 1, (i + 1) * 10000} }] = temp.labels + 1
    end

    trainData.data = trainData.data[{ {1, maxLoad} }]
    trainData.labels = trainData.labels[{ {1, maxLoad} }]

    trainData.data = trainData.data:reshape(maxLoad, 3, geometry[1], geometry[2])

    function trainData:size()
        return trainData.data:size(1)
    end

    return trainData
end

function cifar10.loadTestSet(maxLoad, geometry)
    cifar10.download()

    local testData = {
        data = torch.Tensor(10000, 3072),
        labels = torch.Tensor(10000)
    }

    local temp = torch.load(paths.concat(cifar10.path_folder, 'test_batch.t7'), 'ascii')
    testData.data[{{}}] = temp.data:t()
    testData.labels[{{}}] = temp.labels:squeeze() + 1

    testData.data = testData.data[{ {1, maxLoad} }]
    testData.labels = testData.labels[{ {1, maxLoad} }]

    testData.data = testData.data:reshape(maxLoad, 3, geometry[1], geometry[2])

    function testData:size()
        return testData.data:size(1)
    end

    return testData
end
