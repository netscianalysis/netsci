%rename(generalizedCorrelation)
netcalc::generalizedCorrelation(
        const std::unique_ptr<CuArray<float>> &X,
        const std::unique_ptr<CuArray<float>> &R,
        const std::unique_ptr<CuArray<int>> &ab,
        int k,
        int n,
        int xd,
        int d,
        int platform
);

%rename(generalizedCorrelationWithCheckpointing)
netcalc::generalizedCorrelation(
                        const std::unique_ptr<CuArray<float>> &X,
        const std::unique_ptr<CuArray<float>> &R,
        const std::unique_ptr<CuArray<int>> &ab,
        int k,
        int n,
        int xd,
        int d,
        int platform,
        int checkpointFrequency,
        std::string checkpointFileName
);

%rename(mutualInformation)
netcalc::mutualInformation(
const std::unique_ptr<CuArray<float>> &X,
const std::unique_ptr<CuArray<float>> &I,
const std::unique_ptr<CuArray<int>> &ab,
int k,
int n,
int xd,
int d,
int platform
);

%rename(mutualInformationWithCheckpointing)
netcalc::mutualInformation(
const std::unique_ptr<CuArray<float>> &X,
const std::unique_ptr<CuArray<float>> &I,
const std::unique_ptr<CuArray<int>> &ab,
int k,
int n,
int xd,
int d,
int platform,
int checkpointFrequency,
        std::string checkpointFileName
);

