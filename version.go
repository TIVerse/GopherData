package gopherdata

// Version is the current version of GopherData
const Version = "1.0.0"

// VersionInfo contains detailed version information
type VersionInfo struct {
	Version    string
	GoVersion  string
	CommitHash string
	BuildDate  string
}

// GetVersion returns the current version
func GetVersion() string {
	return Version
}

// GetVersionInfo returns detailed version information
func GetVersionInfo() VersionInfo {
	return VersionInfo{
		Version:    Version,
		GoVersion:  "1.21+",
		CommitHash: "",  // Populated during build
		BuildDate:  "",  // Populated during build
	}
}
