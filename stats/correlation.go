package stats

import (
	"fmt"
	"math"
	"sort"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// Pearson calculates the Pearson correlation coefficient between x and y.
// Returns value in [-1, 1] where:
// 1 = perfect positive correlation
// 0 = no correlation
// -1 = perfect negative correlation
func Pearson(x, y []float64) (float64, error) {
	if len(x) != len(y) {
		return 0, fmt.Errorf("length mismatch: len(x)=%d, len(y)=%d", len(x), len(y))
	}
	if len(x) < 2 {
		return 0, fmt.Errorf("need at least 2 values")
	}
	
	meanX := Mean(x)
	meanY := Mean(y)
	
	var sumXY, sumX2, sumY2 float64
	for i := 0; i < len(x); i++ {
		dx := x[i] - meanX
		dy := y[i] - meanY
		sumXY += dx * dy
		sumX2 += dx * dx
		sumY2 += dy * dy
	}
	
	if sumX2 == 0 || sumY2 == 0 {
		return 0, fmt.Errorf("zero variance")
	}
	
	return sumXY / math.Sqrt(sumX2*sumY2), nil
}

// Spearman calculates the Spearman rank correlation coefficient.
func Spearman(x, y []float64) (float64, error) {
	if len(x) != len(y) {
		return 0, fmt.Errorf("length mismatch")
	}
	
	// Convert to ranks
	rankX := rank(x)
	rankY := rank(y)
	
	// Pearson correlation of ranks
	return Pearson(rankX, rankY)
}

// rank converts values to ranks (average rank for ties).
func rank(values []float64) []float64 {
	n := len(values)
	
	// Create index-value pairs
	type pair struct {
		index int
		value float64
	}
	pairs := make([]pair, n)
	for i, v := range values {
		pairs[i] = pair{i, v}
	}
	
	// Sort by value
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].value < pairs[j].value
	})
	
	// Assign ranks (handling ties with average rank)
	ranks := make([]float64, n)
	i := 0
	for i < n {
		j := i
		// Find end of tie group
		for j < n && pairs[j].value == pairs[i].value {
			j++
		}
		
		// Average rank for tie group
		avgRank := float64(i+j+1) / 2.0 // +1 because ranks start at 1
		for k := i; k < j; k++ {
			ranks[pairs[k].index] = avgRank
		}
		i = j
	}
	
	return ranks
}

// Kendall calculates the Kendall tau correlation coefficient.
// Measures the ordinal association between two quantities.
func Kendall(x, y []float64) (float64, error) {
	if len(x) != len(y) {
		return 0, fmt.Errorf("length mismatch")
	}
	
	n := len(x)
	if n < 2 {
		return 0, fmt.Errorf("need at least 2 values")
	}
	
	concordant := 0
	discordant := 0
	
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			signX := sign(x[j] - x[i])
			signY := sign(y[j] - y[i])
			
			if signX*signY > 0 {
				concordant++
			} else if signX*signY < 0 {
				discordant++
			}
			// Ties contribute 0
		}
	}
	
	total := n * (n - 1) / 2
	return float64(concordant-discordant) / float64(total), nil
}

func sign(x float64) int {
	if x > 0 {
		return 1
	} else if x < 0 {
		return -1
	}
	return 0
}

// Covariance calculates the sample covariance between x and y.
func Covariance(x, y []float64) float64 {
	if len(x) != len(y) || len(x) < 2 {
		return 0
	}
	
	n := float64(len(x))
	meanX := Mean(x)
	meanY := Mean(y)
	
	var sum float64
	for i := 0; i < len(x); i++ {
		sum += (x[i] - meanX) * (y[i] - meanY)
	}
	
	return sum / (n - 1) // Sample covariance
}

// CorrMatrix computes the correlation matrix for all numeric columns in a DataFrame.
func CorrMatrix(df *dataframe.DataFrame, method string) (*dataframe.DataFrame, error) {
	// Get numeric columns
	numCols := getNumericColumns(df)
	if len(numCols) == 0 {
		return nil, fmt.Errorf("no numeric columns found")
	}
	
	n := len(numCols)
	corrData := make(map[string]any)
	
	// Compute correlations
	for _, col1 := range numCols {
		series1, _ := df.Column(col1)
		vals1 := seriesToFloat64(series1)
		
		colData := make([]any, n)
		for j, col2 := range numCols {
			if col1 == col2 {
				colData[j] = 1.0
				continue
			}
			
			series2, _ := df.Column(col2)
			vals2 := seriesToFloat64(series2)
			
			var corr float64
			var err error
			
			switch method {
			case "pearson":
				corr, err = Pearson(vals1, vals2)
			case "spearman":
				corr, err = Spearman(vals1, vals2)
			case "kendall":
				corr, err = Kendall(vals1, vals2)
			default:
				return nil, fmt.Errorf("unknown method: %s", method)
			}
			
			if err != nil {
				colData[j] = nil
			} else {
				colData[j] = corr
			}
		}
		corrData[col1] = colData
	}
	
	// Create result DataFrame
	return dataframe.New(corrData)
}

// CovMatrix computes the covariance matrix for all numeric columns.
func CovMatrix(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	numCols := getNumericColumns(df)
	if len(numCols) == 0 {
		return nil, fmt.Errorf("no numeric columns found")
	}
	
	n := len(numCols)
	covData := make(map[string]any)
	
	for _, col1 := range numCols {
		series1, _ := df.Column(col1)
		vals1 := seriesToFloat64(series1)
		
		colData := make([]any, n)
		for j, col2 := range numCols {
			series2, _ := df.Column(col2)
			vals2 := seriesToFloat64(series2)
			
			cov := Covariance(vals1, vals2)
			colData[j] = cov
		}
		covData[col1] = colData
	}
	
	return dataframe.New(covData)
}

// Helper functions

func getNumericColumns(df *dataframe.DataFrame) []string {
	numCols := make([]string, 0)
	
	for _, col := range df.Columns() {
		series, err := df.Column(col)
		if err != nil {
			continue
		}
		
		// Check if column is numeric
		if isNumericSeriesStats(series) {
			numCols = append(numCols, col)
		}
	}
	
	return numCols
}

func isNumericSeriesStats(s interface{ Len() int; Get(int) (any, bool); Dtype() core.Dtype }) bool {
	dtype := s.Dtype()
	switch dtype {
	case core.DtypeFloat64, core.DtypeInt64:
		return true
	}
	return false
}

func seriesToFloat64(s *seriesPkg.Series[any]) []float64 {
	result := make([]float64, 0, s.Len())
	
	for i := 0; i < s.Len(); i++ {
		val, ok := s.Get(i)
		if !ok || val == nil {
			continue
		}
		
		switch v := val.(type) {
		case float64:
			result = append(result, v)
		case float32:
			result = append(result, float64(v))
		case int:
			result = append(result, float64(v))
		case int64:
			result = append(result, float64(v))
		case int32:
			result = append(result, float64(v))
		case int16:
			result = append(result, float64(v))
		case int8:
			result = append(result, float64(v))
		}
	}
	
	return result
}
