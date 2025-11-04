package hypothesis

import (
	"fmt"

	"gonum.org/v1/gonum/stat/distuv"
)

// ChiSquareResult contains the results of a chi-square test.
type ChiSquareResult struct {
	Statistic float64 // chi-square statistic
	PValue    float64 // p-value
	DF        int     // degrees of freedom
}

// ChiSquare performs a chi-square test for independence.
// observed and expected are contingency tables (2D slices).
func ChiSquare(observed, expected [][]float64) ChiSquareResult {
	if len(observed) == 0 || len(observed[0]) == 0 {
		return ChiSquareResult{}
	}
	
	if len(observed) != len(expected) || len(observed[0]) != len(expected[0]) {
		return ChiSquareResult{}
	}
	
	rows := len(observed)
	cols := len(observed[0])
	
	// Calculate chi-square statistic
	chiSq := 0.0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if expected[i][j] > 0 {
				diff := observed[i][j] - expected[i][j]
				chiSq += (diff * diff) / expected[i][j]
			}
		}
	}
	
	// Degrees of freedom
	df := (rows - 1) * (cols - 1)
	
	// Calculate p-value
	chiDist := distuv.ChiSquared{K: float64(df)}
	pValue := 1 - chiDist.CDF(chiSq)
	
	return ChiSquareResult{
		Statistic: chiSq,
		PValue:    pValue,
		DF:        df,
	}
}

// ChiSquareGOF performs a chi-square goodness-of-fit test.
// Tests whether the observed frequencies match the expected frequencies.
func ChiSquareGOF(observed, expected []float64) ChiSquareResult {
	if len(observed) != len(expected) || len(observed) == 0 {
		return ChiSquareResult{}
	}
	
	// Calculate chi-square statistic
	chiSq := 0.0
	for i := range observed {
		if expected[i] > 0 {
			diff := observed[i] - expected[i]
			chiSq += (diff * diff) / expected[i]
		}
	}
	
	// Degrees of freedom
	df := len(observed) - 1
	
	// Calculate p-value
	chiDist := distuv.ChiSquared{K: float64(df)}
	pValue := 1 - chiDist.CDF(chiSq)
	
	return ChiSquareResult{
		Statistic: chiSq,
		PValue:    pValue,
		DF:        df,
	}
}

// ChiSquareIndependence performs a chi-square test of independence from a contingency table.
// Calculates expected frequencies automatically from observed data.
func ChiSquareIndependence(observed [][]float64) (ChiSquareResult, error) {
	if len(observed) == 0 || len(observed[0]) == 0 {
		return ChiSquareResult{}, fmt.Errorf("empty contingency table")
	}
	
	rows := len(observed)
	cols := len(observed[0])
	
	// Calculate row and column totals
	rowTotals := make([]float64, rows)
	colTotals := make([]float64, cols)
	total := 0.0
	
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			rowTotals[i] += observed[i][j]
			colTotals[j] += observed[i][j]
			total += observed[i][j]
		}
	}
	
	// Calculate expected frequencies
	expected := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		expected[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			expected[i][j] = (rowTotals[i] * colTotals[j]) / total
		}
	}
	
	return ChiSquare(observed, expected), nil
}
