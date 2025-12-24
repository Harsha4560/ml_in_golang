package layers

import (
	"nnscratch/tensor"
)

type Parameter struct {
	Value *tensor.Tensor
	Grad *tensor.Tensor
}

type Layer interface {
	Forward(input *tensor.Tensor) (*tensor.Tensor, error)
	Backward(gradOutput *tensor.Tensor, lr float64) (*tensor.Tensor, error)
	GetParameters() []*Parameter
	GetWeights() []*tensor.Tensor
	GetBiases() []*tensor.Tensor
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

func (s *Sequential) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
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

func (s *Sequential) Backward(grad *tensor.Tensor, lr float64) error {
	var err error
	for i := len(s.Layers) - 1; i >= 0; i-- {
		grad, err = s.Layers[i].Backward(grad, lr)
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *Sequential) GetParameters() []*Parameter {
	var params []*Parameter 
	for _, layer := range(s.Layers) {
		params = append(params, layer.GetParameters()...)
	}
	return params
}

func (s *Sequential) GetWeights() []*tensor.Tensor {
	var weights []*tensor.Tensor
	for _, layer := range(s.Layers) {
		weights = append(weights, layer.GetWeights()...)
	}
	return weights
}

func (s *Sequential) GetBiases() []*tensor.Tensor {
	var biases []*tensor.Tensor
	for _, layer := range(s.Layers) {
		biases = append(biases, layer.GetBiases()...)
	}
	return biases
}