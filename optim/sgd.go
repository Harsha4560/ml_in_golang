package optim

import (
	"nnscratch/layers"
	"nnscratch/tensor"
)

type Optimizer interface {
	Step() error
	ZeroGrad() 
}

type SGD struct {
	Parameters []*layers.Parameter 
	LR float64
}

func NewSGD(params []*layers.Parameter, lr float64) *SGD {
	return &SGD{
		Parameters: params,
		LR: lr,
	}
}

func (s *SGD) Step() error {
	for _, p := range s.Parameters {
		if p.Grad == nil {
			continue
		}
		update, err := p.Grad.MulScalar(s.LR)
		if err != nil {
			return err
		}
		p.Value, err = tensor.TensorDiff(p.Value, update)
		if err != nil {
			return err
		}
	}
	return nil 
}

func (s *SGD) ZeroGrad() {
	for _, p := range s.Parameters {
		if p.Grad != nil {
			for i := range p.Grad.Data() {
				p.Grad.Data()[i] = 0
			}
		}
	}
}
