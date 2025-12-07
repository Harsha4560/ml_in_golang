package main

import (
	"math/rand"
)

type Layer interface {
	Forward(input *Tensor) (*Tensor, error)
	Backward(gradOutput *Tensor, lr float64) (*Tensor, error)
}

type LossLayer interface {
	Loss(y_pred *Tensor, y_actual *Tensor) (float64, error)
	Diffrential() (*Tensor, error)
}

type Sequential struct {
	Layers    []Layer
	LossLayer LossLayer
}

func NewSequential(layers ...Layer) *Sequential {
	return &Sequential{
		Layers: layers,
	}
}

func (s *Sequential) Forward(input *Tensor) (*Tensor, error) {
	out := input
	var err error
	for _, layer := range s.Layers {
		out, err = layer.Forward(out)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (s *Sequential) Backward(grad *Tensor, lr float64) error {
	var err error
	for i := len(s.Layers) - 1; i >= 0; i-- {
		grad, err = s.Layers[i].Backward(grad, lr)
		if err != nil {
			return err
		}
	}
	return nil
}

// Loss functions

// Mean square error loss
type MSELossLayer struct {
	y_pred   *Tensor
	y_actual *Tensor
}

func (l *MSELossLayer) Loss(y_pred *Tensor, y_actual *Tensor) (float64, error) {
	l.y_actual = y_actual
	l.y_pred = y_pred
	res, err := MSE(y_pred, y_actual)
	if err != nil {
		return 0, err
	}
	return res, nil
}

func (l *MSELossLayer) Diffrential() (*Tensor, error) {
	res, err := DiffMSE(l.y_pred, l.y_actual)
	if err != nil {
		return nil, err
	}
	return res, nil
}

// BCE loss layer should always have a sigmoid function before it to make sure the input is less than 1 
type BCELossLayer struct {
	y_pred *Tensor
	y_actual *Tensor
}

func (l *BCELossLayer) Loss(y_pred *Tensor, y_actual *Tensor) (float64, error) {
	l.y_actual = y_actual 
	l.y_pred = y_pred 
	res, err :=BCE(y_pred, y_actual) 
	if err != nil {
		return 0, err
	}
	return res, nil
}

func (l *BCELossLayer) Diffrential() (*Tensor, error) {
	res, err := DiffBCE(l.y_pred, l.y_actual)
	if err != nil {
		return nil, err
	}
	return res, nil
}

// Activation functions

// Sigmoid Function
type SigmoidLayer struct {
	output *Tensor
}

func (s *SigmoidLayer) Forward(input *Tensor) (*Tensor, error) {
	res, err := input.Apply("Sigmoid")
	if err != nil {
		return nil, err
	}
	s.output = res
	return res, nil
}

func (s *SigmoidLayer) Backward(gradOutput *Tensor, lr float64) (*Tensor, error) {
	ones, _ := NewTensorOnes(gradOutput.shape...)
	oneminusy, err := TensorDiff(ones, s.output)
	if err != nil {
		return nil, err
	}
	diffren, _ := TensorMul(s.output, oneminusy)
	return TensorMul(gradOutput, diffren)
}

type DenseLayer struct {
	Weights *Tensor
	Bias    *Tensor
	input   *Tensor
}

func NewDenseLayer(in_features int, out_features int) *DenseLayer {
	w, _ := NewTensor(out_features, in_features)
	b, _ := NewTensor(out_features)

	// Initlialize to random numbers
	for i := range w.data {
		w.data[i] = rand.NormFloat64() * 0.01
	}
	for i := range b.data {
		b.data[i] = rand.NormFloat64() * 0.01
	}

	return &DenseLayer{
		Weights: w,
		Bias:    b,
	}

}

func (d *DenseLayer) Forward(input *Tensor) (*Tensor, error) {
	d.input = input
	w_t, _ := d.Weights.Transpose()
	res, err := Matmul(input, w_t)
	if err != nil {
		return nil, err
	}
	return res, nil
}

func (d *DenseLayer) Backward(gradOutput *Tensor, lr float64) (*Tensor, error) {
	go_t, _ := gradOutput.Transpose()
	dW, err := Matmul(go_t, d.input)
	if err != nil {
		return nil, err
	}
	stepW, _ := dW.MulScalar(lr)

	// calculate gradient to return dl/dx
	gradInput, _ := Matmul(gradOutput, d.Weights)

	//Update the weights
	d.Weights, err = TensorDiff(d.Weights, stepW)
	if err != nil {
		return nil, err
	}

	batchSize := gradOutput.shape[0]
	onesVector, err := NewTensorOnes(1, batchSize)
	if err != nil {
		return nil, err
	}
	db_raw, err := Matmul(onesVector, gradOutput)
	if err != nil {
		return nil, err
	}

	// Reshape to bias shape
	db_adjusted, err := db_raw.Reshape(d.Bias.shape...)
	if err != nil {
		return nil, err
	}

	stepB, _ := db_adjusted.MulScalar(lr)
	d.Bias, err = TensorDiff(d.Bias, stepB)
	if err != nil {
		return nil, err
	}

	return gradInput, nil
}
