package layers

import (
	"nnscratch/tensor"
	"nnscratch/activations"
	"nnscratch/maths"
)

// Sigmoid Function
type SigmoidLayer struct {
	output *tensor.Tensor
}

func (s *SigmoidLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	res, err := input.Apply(activations.Sigmoid)
	if err != nil {
		return nil, err
	}
	s.output = res
	return res, nil
}

func (s *SigmoidLayer) Backward(gradOutput *tensor.Tensor, lr float64) (*tensor.Tensor, error) {
	ones, _ := tensor.NewTensorOnes(gradOutput.Shape()...)
	oneminusy, err := tensor.TensorDiff(ones, s.output)
	if err != nil {
		return nil, err
	}
	diffren, _ := tensor.TensorMul(s.output, oneminusy)
	return tensor.TensorMul(gradOutput, diffren)
}

func (s *SigmoidLayer) GetParameters() []*Parameter {
	return []*Parameter{}
}

func (s *SigmoidLayer) GetWeights() []*tensor.Tensor {
	return nil
}

func (s *SigmoidLayer) GetBiases() []*tensor.Tensor {
	return nil
}



// Sine and cosine layers
type SineLayer struct {
	input *tensor.Tensor
}

func (s *SineLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	s.input = input
	return input.Apply(maths.Sin)
}

func (s *SineLayer) Backward(gradOutput *tensor.Tensor, lr float64) (*tensor.Tensor, error) {
	diffren, err := s.input.Apply(maths.Cos)
	if err != nil {
		return nil, err
	}
	return tensor.TensorMul(gradOutput, diffren)
}

func (s *SineLayer) GetParameters() []*Parameter {
	return []*Parameter{}
}

func (s *SineLayer) GetWeights() []*tensor.Tensor {
	return nil 
}

func (s *SineLayer) GetBiases() []*tensor.Tensor {
	return nil 
}

type CosineLayer struct {
	input *tensor.Tensor
}

func (s *CosineLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	s.input = input
	return input.Apply(maths.Cos)
}

func (s *CosineLayer) Backward(gradOutput *tensor.Tensor, lr float64) (*tensor.Tensor, error) {
	diffren, err := s.input.Apply(maths.Sin)
	if err != nil {
		return nil, err
	}
	diffren, _ = diffren.MulScalar(-1.0)
	return tensor.TensorMul(gradOutput, diffren)
}

func (s *CosineLayer) GetParameters() []*Parameter {
	return []*Parameter{}
}

func (s *CosineLayer) GetWeights() []*tensor.Tensor {
	return nil 
}

func (s *CosineLayer) GetBiases() []*tensor.Tensor {
	return nil 
}
