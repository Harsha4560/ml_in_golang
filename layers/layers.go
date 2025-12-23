package layers

import (
	"nnscratch/activations"
	"nnscratch/loss"
	"nnscratch/maths"
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
}

type LossLayer interface {
	Loss(y_pred *tensor.Tensor, y_actual *tensor.Tensor) (float64, error)
	Diffrential() (*tensor.Tensor, error)
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

// Loss functions

// Mean square error loss
type MSELossLayer struct {
	y_pred   *tensor.Tensor
	y_actual *tensor.Tensor
}

func (l *MSELossLayer) Loss(y_pred *tensor.Tensor, y_actual *tensor.Tensor) (float64, error) {
	l.y_actual = y_actual
	l.y_pred = y_pred
	res, err := loss.MSE(y_pred, y_actual)
	if err != nil {
		return 0, err
	}
	return res, nil
}

func (l *MSELossLayer) Diffrential() (*tensor.Tensor, error) {
	res, err := loss.DiffMSE(l.y_pred, l.y_actual)
	if err != nil {
		return nil, err
	}
	return res, nil
}

// BCE loss layer should always have a sigmoid function before it to make sure the input is less than 1
type BCELossLayer struct {
	y_pred   *tensor.Tensor
	y_actual *tensor.Tensor
}

func (l *BCELossLayer) Loss(y_pred *tensor.Tensor, y_actual *tensor.Tensor) (float64, error) {
	l.y_actual = y_actual
	l.y_pred = y_pred
	res, err := loss.BCE(y_pred, y_actual)
	if err != nil {
		return 0, err
	}
	return res, nil
}

func (l *BCELossLayer) Diffrential() (*tensor.Tensor, error) {
	res, err := loss.DiffBCE(l.y_pred, l.y_actual)
	if err != nil {
		return nil, err
	}
	return res, nil
}

// Activation functions

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

type DenseLayer struct {
	Weights *Parameter
	Bias    *Parameter
	input   *tensor.Tensor
}

func NewDenseLayer(in_features int, out_features int) *DenseLayer {
	w, _ := tensor.NewTensorRandom(out_features, in_features)
	b, _ := tensor.NewTensorRandom(1, out_features)

	w_grad, _ := tensor.NewTensor(out_features, in_features)
	b_grad, _ := tensor.NewTensor(1, out_features)

	return &DenseLayer{
		Weights: &Parameter{Value: w, Grad: w_grad},
		Bias:    &Parameter{Value: b, Grad: b_grad},
	}

}

func (d *DenseLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	d.input = input
	w_t, _ := d.Weights.Value.Transpose()
	res, err := tensor.Matmul(input, w_t)
	if err != nil {
		return nil, err
	}
	res, err = tensor.TensorAdd(res, d.Bias.Value)
	if err != nil {
		return nil, err
	}
	return res, nil
}

func (d *DenseLayer) Backward(gradOutput *tensor.Tensor, lr float64) (*tensor.Tensor, error) {
	go_t, _ := gradOutput.Transpose()
	dW, err := tensor.Matmul(go_t, d.input)
	if err != nil {
		return nil, err
	}
	d.Weights.Grad = dW

	gradInput, _ := tensor.Matmul(gradOutput, d.Weights.Value)

	batchSize := gradOutput.Shape()[0]
	onesVector, err := tensor.NewTensorOnes(1, batchSize)
	if err != nil {
		return nil, err
	}
	db_raw, err := tensor.Matmul(onesVector, gradOutput)
	if err != nil {
		return nil, err
	}

	// Reshape to bias shape
	db_adjusted, err := db_raw.Reshape(d.Bias.Value.Shape()...)
	if err != nil {
		return nil, err
	}
	d.Bias.Grad = db_adjusted
	return gradInput, nil
}

func (d *DenseLayer) GetParameters() []*Parameter {
	return []*Parameter{d.Weights, d.Bias}
}
