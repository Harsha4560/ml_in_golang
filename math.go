package main

import (
	"fmt"
	"strings"
)

func Power(base float64, pow float64) (float64, error) {
	numerator, denominator, err := Fraction(pow)
	if err != nil {
		return 0, err
	}
	root := nthRoot(base, denominator)
	result := powerInt(root, numerator)
	return result, nil
}

func powerInt(x float64, y int64) float64 {
	if y == 0 {
		return 1.0
	}
	if y < 0 {
		x = 1 / x
		y = -y
	}
	result := 1.0
	for y > 0 {
		result *= x
		y--
	}
	return result
}

// use Newton raphson method for estimatig the nth root
func nthRoot(x float64, n int64) float64 {
	if n == 0 {
		return 1.0
	}
	if x == 0 {
		return 0.0
	}
	guess := 1.0
	if x > 1 {
		guess = x / float64(n)
	}
	for i := 0; i < 1000; i++ {
		prevGuess := guess
		guess = (float64(n-1)*prevGuess + (x / powerInt(prevGuess, n-1))) / float64(n)
		if guess == prevGuess {
			break
		}
	}
	return guess
}

func Fraction(f float64) (num, den int64, err error) {
	if f == 0 {
		return 0, 1, nil
	}
	var sign int64 = 1
	if f < 0 {
		sign = -1
		f = -f
	}

	intPart := int64(f)
	fracPart := f - float64(intPart)

	num, den, err = convertFrac(fracPart)
	if err != nil {
		return 0, 0, err
	}
	num += intPart * den
	return num * sign, den, nil
}

func convertFrac(f float64) (num, den int64, err error) {
	s := fmt.Sprintf("%.12f", f)
	s = strings.TrimPrefix(s, "0.")

	var denominator int64 = 1
	for range s {
		denominator *= 10
		if denominator < 0 {
			return 0, 0, fmt.Errorf("convertFrac: denominator overflow for %f", f)
		}
	}
	var numerator int64
	fmt.Sscanf(s, "%d", &numerator)

	common := Gcd(numerator, denominator)
	return numerator / common, denominator / common, nil
}

func Gcd(a, b int64) int64 {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func Factorial(n int) float64 {
	if n <= 1 {
		return 1
	}
	result := 1.0
	for i:=1; i<=n; i++ {
		result *= float64(i)
	}
	return result
}

// Absolute value function 
func Abs(x float64) float64 {
	if x >= 0 {return x}
	return -x
}

// this is using the taylor serires expansion 
func Exp(x float64) (float64, error) {
	result := 1.0
	epsilon := 1e-15
	for n:=1; ; n++ {
		pow, err := Power(x, float64(n))
		if err != nil {
			return 0, err
		}
		newResult := result + (pow/Factorial(n))
		if Abs(result - newResult) <= epsilon {
			return newResult, nil 
		}
		result = newResult
	}
}


