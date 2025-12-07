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
			delLoss, err = delLoss.Reshape(1, 1)
			if err != nil {
				return err
			}
			delW, err := Matmul(x, delLoss)
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
	Bias    float64
}

func NewLogisticRegression(numFeatures int) *LogisticRegression {
	weights, _ := NewTensor(numFeatures)
	for i := range weights.data {
		weights.data[i] = rand.NormFloat64() * 0.01
	}
	return &LogisticRegression{
		Weights: weights,
		Bias:    rand.NormFloat64() * 0.01,
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

func (l *LogisticRegression) Train(X *Tensor, Y *Tensor, lr float64, epochs int) error {
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
	for i := range Y.data {
		if Y.data[i] != 1.0 && Y.data[i] != 0.0 {
			return fmt.Errorf("train: The Y tensor has a non 0 or 1 val at %v. only 1 or 0 should be there in logistic regression it is %v", i, Y.data[i])
		}
	}

	for epoch := 0; epoch < epochs; epoch++ {
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
			y_pred, err := l.Predict(x)
			if err != nil {
				return err
			}
			loss, err := BCE(y, y_pred)
			if err != nil {
				return err
			}
			sumloss += loss
			difflbysigma, err := DiffBCE(y, y_pred)
			if err != nil {
				return err
			}
			// Calculate derivative of sigmoid: y_pred * (1 - y_pred)
			// We cannot use Apply("DiffSigmoid") on difflbysigma because that would
			// calculate the derivative of the *gradient values*, not the predictions.
			ones, _ := NewTensorOnes(y_pred.shape...)
			oneMinusYPred, _ := TensorDiff(ones, y_pred)
			sigmoidDeriv, _ := TensorMul(y_pred, oneMinusYPred)

			diffsigma, err := TensorMul(difflbysigma, sigmoidDeriv)
			if err != nil {
				return err
			}
			diffsigma, err = diffsigma.Reshape(1, 1)
			if err != nil {
				return err
			}
			x_new, err := x.Reshape(1, 2)
			if err != nil {
				return err
			}
			delW, err := Matmul(diffsigma, x_new)
			if err != nil {
				return err
			}
			delW, _ = delW.MulScalar(lr)
			delW, err = delW.Reshape(l.Weights.shape...)
			if err != nil {
				return err
			}
			l.Weights, err = TensorDiff(l.Weights, delW)
			if err != nil {
				return err
			}
			delB, err := diffsigma.MulScalar(lr)
			if err != nil {
				return err
			}
			db, _ := delB.Get(0)
			l.Bias = l.Bias - db
		}
		fmt.Printf("Loss: %v | epoch: %v\n", sumloss/float64(len(X.data)), epoch)
	}
	return nil

}
