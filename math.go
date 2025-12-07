package main

import (
	"fmt"
	"math"
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

func Round(a float64) float64 {
	if a-float64(int(a)) >= 0.5 {
		return float64(int(a)) + 1
	}
	return float64(int(a))
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
	s := fmt.Sprintf("%.5f", f)
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
	for i := 1; i <= n; i++ {
		result *= float64(i)
	}
	return result
}

// Absolute value function
func Abs(x float64) float64 {
	if x >= 0 {
		return x
	}
	return -x
}

// this is using the taylor serires expansion
func Exp(x float64) float64 {
	return math.Exp(x)
}

const pi = 3.141592653589793

// converts radians to degrees
func Degrees(x float64) float64 {
	return (x * 180.0) / pi
}

// converts degrees to radians
func Radians(x float64) float64 {
	return (x * pi) / 180.0
}

// Calculate the sine function using the Taylor series
// Takes input in radians
func Sin(x float64) float64 {
	for x >= 2*pi {
		x -= 2 * pi
	}
	epsilon := 1e-15
	result := 0.0
	flag := 1.0
	for n := 1; ; n += 2 {
		pow, _ := Power(x, float64(n))
		newResult := result + flag*(pow/Factorial(n))
		if Abs(result-newResult) <= epsilon {
			return newResult
		}
		result = newResult
		flag *= -1
	}
}

func Cos(x float64) float64 {
	for x >= 2*pi {
		x -= 2 * pi
	}
	epsilon := 1e-15
	result := 0.0
	for n := 0; ; n++ {
		pow, _ := Power(x, float64(2*n))
		flag, _ := Power(-1, float64(n))
		newResult := result + flag*(pow/Factorial(2*n))
		if Abs(result-newResult) <= epsilon {
			return newResult
		}
		result = newResult
	}
}

// Using newton raphson method
func Ln(x float64) float64 {
	if math.IsNaN(x) {
		return x
	}
	if x <= 0 {
		return math.NaN()
	}

	if x == 1 {
		return 0.0
	}

	guess := 1.0
	epsilon := 1e-15

	for i := 0; i < 2000; i++ {
		newGuess := guess - 1 + (x / Exp(guess))
		if Abs(newGuess-guess) <= epsilon {
			return newGuess
		}
		guess = newGuess
	}
	return guess

}

func Log(x float64, base float64) float64 {
	return Ln(x) / Ln(base)
}
