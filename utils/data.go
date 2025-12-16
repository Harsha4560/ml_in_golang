package utils

import (
	"math/rand"
	"nnscratch/tensor"
)

type DataLoader struct {
	inputs *tensor.Tensor
	targets *tensor.Tensor
	batchSize int
	shuffle bool 
	ignoreLast bool 
}

func NewDataLoader(inputs, targets *tensor.Tensor, batchSize int, shuffle bool) *DataLoader {
	return &DataLoader{
		inputs: inputs, 
		targets: targets,
		batchSize: batchSize,
		shuffle: shuffle,
		ignoreLast: false,
	}
}

type BatchIterator struct {
	loader *DataLoader
	indices []int
	current int
	currnetIndices []int
}

func (dl *DataLoader) MakeIterator() *BatchIterator {
	n := dl.inputs.Shape()[0]
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i 
	}
	if dl.shuffle {
		rand.Shuffle(n, func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}
	return &BatchIterator{
		loader: dl,
		indices: indices,
		current: 0,
	}
}

func (it *BatchIterator) Next() bool {
	if it.current >= len(it.indices) {
		return false 
	}

	end := it.current + it.loader.batchSize
	if it.loader.ignoreLast && end > len(it.indices) {
		return false
	}
	if end > len(it.indices) {
		end = len(it.indices)
	}
	it.currnetIndices = it.indices[it.current:end]
	it.current = end
	return true 
}

func (it *BatchIterator) Get() (*tensor.Tensor, *tensor.Tensor) {
	xBatch, err := it.loader.inputs.GetBatchElements(it.currnetIndices)
	if err != nil {
		panic(err)
	}
	var yBatch *tensor.Tensor
	if it.loader.targets != nil {
		yBatch, err = it.loader.targets.GetBatchElements(it.currnetIndices)
		if err != nil {
			panic(err)
		}
	}
	return xBatch, yBatch
}



