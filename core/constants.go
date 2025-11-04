package core

// Version is the current version of GopherData.
const Version = "v1.0.0"

// DefaultWorkers specifies the default number of parallel workers.
// 0 means use runtime.NumCPU().
var DefaultWorkers = 0

// DefaultNAValues is the default list of strings treated as null/NA values.
var DefaultNAValues = []string{
	"",
	"NA",
	"N/A",
	"NULL",
	"null",
	"NaN",
	"nan",
	"#N/A",
	"#NA",
	"None",
	"none",
	"-",
}
