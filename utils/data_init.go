package utils

import (
	"math/rand"
	"nnscratch/tensor"
)

func GenerateNormalData(t *tensor.Tensor, mean float64, stdDev float64) {
	for i := range t.Data() {
		t.Data()[i] = (rand.NormFloat64() * stdDev) + mean
	}
}

