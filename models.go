package main

import (
	"fmt"
	"math/rand"
	// "time"
)

// LINEARR REGRESSION
type LinearRegression struct {
	Weights *Tensor
	Bias    float64
}

func NewLinearRegression(numFeatures int) *LinearRegression {
	weights, _ := NewTensor(numFeatures)
	for i := range weights.data {
		weights.data[i] = rand.NormFloat64() * 0.01
	}
	return &LinearRegression{
		Weights: weights,
		Bias:    rand.NormFloat64() * 0.01,
	}
}

func (lr *LinearRegression) Predict(X *Tensor) (*Tensor, error) {
	if len(X.shape) != 1 {
		return nil, fmt.Errorf("predict: the input shape is not 1d")
	}
	if X.shape[0] != lr.Weights.shape[0] {
		return nil, fmt.Errorf("predict: The input is not the same shape as features %v, %v", X.shape, lr.Weights.shape)
	}
	a, _ := TensorMul(X, lr.Weights)
	b, _ := a.Sum()
	b, _ = b.AddScalar(lr.Bias)
	return b, nil
}

func (l *LinearRegression) Train(X *Tensor, Y *Tensor, lr float64, epochs int) error {
	// check for shapes of the input tensors
	if len(X.shape) != 2 {
		return fmt.Errorf("train: the length of input X tensor shape is not 2 %v", X.shape)
	}
	if len(Y.shape) != 2 {
		return fmt.Errorf("train: The length of input Y tensor shape is not 2 %v", Y.shape)
	}
	if Y.shape[1] != 1 {
		return fmt.Errorf("train: The Y tensor should have a shape (m, 1) it has %v", Y.shape[1])
	}
	if X.shape[0] != Y.shape[0] {
		return fmt.Errorf("train: Both X and Y tensor should be (m, p) and (m, 1) they are %v , %v", X.shape, Y.shape)
	}

	for i := 0; i < epochs; i++ {
		sumloss := 0.0
		for j := range X.shape[0] {
			x, err := X.Slice(j)
			if err != nil {
				return err
			}
			y, err := Y.Slice(j)
			if err != nil {
				return err
			}
			// forward propagation
			y_pred, err := l.Predict(x)
			if err != nil {
				return err
			}
			// calculate loss 
			loss, err := MSE(y, y_pred)
			if err != nil {
				return err
			}
			sumloss += loss
			delLoss, err := DiffMSE(y, y_pred)
			if err != nil {
				return err
			}
			delW, err := TensorMul(delLoss, x)
			if err != nil {
				return err
			}
			delW, _ = delW.MulScalar(-1 * lr)
			// Update weights
			l.Weights, err = TensorDiff(l.Weights, delW)
			if err != nil {
				return err
			}
			// Update bias
			delB, _ := delLoss.MulScalar(lr)
			db, _ := delB.Get(0)
			l.Bias = l.Bias + db
		}
	}
	return nil
}


// Logistic Regression 
type LogisticRegression struct {
	Weights *Tensor
	Bias float64
}

func NewLogisticRegression(numFeatures int) *LogisticRegression {
	weights, _ := NewTensor(numFeatures)
	for i := range(weights.data) {
		weights.data[i] = rand.NormFloat64() * 0.01
	}
	return &LogisticRegression{
		Weights: weights, 
		Bias: rand.NormFloat64() * 0.01,
	}
}

func (lr *LogisticRegression) Predict(X *Tensor) (*Tensor, error) {
	if len(X.shape) != 1 {
		return nil, fmt.Errorf("predict: Input shape is not 1d")
	}
	if X.shape[0] != lr.Weights.shape[0] {
		return nil, fmt.Errorf("predict: Input shape does not match the number of features")
	}
	a, _ := TensorMul(X, lr.Weights)
	b, _ := a.Sum()
	b, _ = b.AddScalar(lr.Bias)
	b, _ = b.Apply("Sigmoid")
	return b, nil
}



