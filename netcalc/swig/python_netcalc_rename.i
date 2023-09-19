%rename(generalizedCorrelation)
netcalc::generalizedCorrelation(
        CuArray<float>* X,
        CuArray<float>* R,
        CuArray<int>* ab,
        int k,
        int n,
        int xd,
        int d,
        int platform
);

%rename(generalizedCorrelationWithCheckpointing)
netcalc::generalizedCorrelation(
        CuArray<float>* X,
        CuArray<float>* R,
        CuArray<int>* ab,
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
        CuArray<float> *X,
CuArray<float> *I,
CuArray<int> *ab,
int k,
int n,
int xd,
int d,
int platform
);

%rename(mutualInformationWithCheckpointing)
netcalc::mutualInformation(
        CuArray<float> *X,
CuArray<float> *I,
CuArray<int> *ab,
int k,
int n,
int xd,
int d,
int platform,
int checkpointFrequency,
        std::string checkpointFileName
);

