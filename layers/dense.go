package layers 

import (
	"nnscratch/tensor"
)

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

func (d *DenseLayer) GetWeights() []*tensor.Tensor {
	return []*tensor.Tensor{d.Weights.Value}
}

func (d *DenseLayer) GetBiases() []*tensor.Tensor {
	return []*tensor.Tensor{d.Bias.Value}
}
