package activations

import (
	"nnscratch/maths"
)

// Sigmoid funtion
func Sigmoid(a float64) float64 {
	e := maths.Exp(-a)
	result := 1.0 / (1.0 + e)
	return result
}

func DiffSigmoid(a float64) float64 {
	sigx := Sigmoid(a)
	return sigx * (1.0 - sigx)
}

// relu function
func ReLu(a float64) float64 {
	if a < 0 {
		return 0
	}
	return a
}

func DiffReLu(a float64) float64 {
	if a < 0 {
		return 0
	}
	return 1.0
}