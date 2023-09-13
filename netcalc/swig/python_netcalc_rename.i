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

%rename(generalizedCorrelationRestartWithCheckpointing)
netcalc::generalizedCorrelation(
        CuArray<float>* X,
CuArray<float>* R,
int k,
int n,
int xd,
int d,
int platform,
int checkpointFrequency,
        std::string checkpointFileName,
const std::string &restartRFileName,
const std::string &restartAbFileName
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

%rename(mutualInformationRestartWithCheckpointing)
netcalc::mutualInformation(
        CuArray<float> *X,
CuArray<float> *I,
int k,
int n,
int xd,
int d,
int platform,
int checkpointFrequency,
        std::string checkpointFileName,
const std::string &restartIFileName,
const std::string &restartAbFileName
);
