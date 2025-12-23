package optim 

import (
	"nnscratch/layers"
	"nnscratch/tensor"
	"nnscratch/maths"
)


type Adam struct {
	Parameters []*layers.Parameter
	LR float64
	Beta_1 float64
	Beta_2 float64
	epsilon float64
	T int 
	M []*tensor.Tensor
	V []*tensor.Tensor
}

func NewAdam(params []*layers.Parameter, lr float64) *Adam {
	m := make([]*tensor.Tensor, len(params))
	v := make([]*tensor.Tensor, len(params))
	for i, p := range params {
		m[i], _ = tensor.NewTensor(p.Value.Shape()...)
		v[i], _ = tensor.NewTensor(p.Value.Shape()...)
	}
	return &Adam{
		Parameters: params,
		LR: lr,
		Beta_1: 0.9,
		Beta_2: 0.99,
		epsilon: 1e-8,
		T: 0,
		M: m,
		V: v,
	}
}

func (a *Adam) Step() error {
	a.T++ 
	for i, p := range a.Parameters {
		term1m, _ := a.M[i].MulScalar(a.Beta_1) 
		term2m, _ := p.Grad.MulScalar(1-a.Beta_1)
		a.M[i], _ = tensor.TensorAdd(term1m, term2m)

		term1v, _ := a.V[i].MulScalar(a.Beta_2)
		grad_square, _ := p.Grad.Power(2)
		term2v, _ := grad_square.MulScalar(1-a.Beta_2)
		a.V[i], _ = tensor.TensorAdd(term1v, term2v)
		
		mcap, _ := a.M[i].MulScalar(1.0/(1 - maths.Power(a.Beta_1, float64(a.T))))
		vcap, _ := a.V[i].MulScalar(1.0/(1 - maths.Power(a.Beta_2, float64(a.T))))
		
		vroot, _ := vcap.Power(0.5)
		vroot, _ = vroot.AddScalar(a.epsilon)
		division, _ := tensor.TensorDiv(mcap, vroot)
		update, err := division.MulScalar(a.LR)
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

func (a *Adam) ZeroGrad() {
		for _, p := range a.Parameters {
		if p.Grad != nil {
			for i := range p.Grad.Data() {
				p.Grad.Data()[i] = 0
			}
		}
	}
}
