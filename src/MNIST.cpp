#include "MNIST.h"

void print_progress(std::string msg, float progress) {
    int barWidth = 70;
    std::cout << msg << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

MNIST::MNIST(std::vector<std::vector<uint8_t>> &tr_x, std::vector<uint8_t> &tr_y, std::vector<std::vector<uint8_t>> &te_x, std::vector<uint8_t> &te_y) 
    : model(RECORD)
{
    train_x.resize(tr_x.size());
    for (size_t i=0; i<tr_x.size(); ++i) {
        train_x[i].resize(tr_x[i].size());
        for (size_t j=0; j<tr_x[i].size(); ++j) {
            train_x[i][j] = 1.0*static_cast<int>(tr_x[i][j])/255;
        }
    }
    train_y.resize(tr_y.size());
    for (size_t i=0; i<tr_y.size(); ++i) {
        train_y[i] = static_cast<int>(tr_y[i]);
    }
    test_x.resize(te_x.size());
    for (size_t i=0; i<te_x.size(); ++i) {
        test_x[i].resize(te_x[i].size());
        for (size_t j=0; j<te_x[i].size(); ++j) {
            test_x[i][j] = 1.0*static_cast<int>(te_x[i][j])/255;
        }
    }
    test_y.resize(te_y.size());
    for (size_t i=0; i<te_y.size(); ++i) {
        test_y[i] = static_cast<int>(te_y[i]);
    }
    /* Create Activation Functions */
    act_list = {std::make_shared<Sigmoid>(), std::make_shared<Softmax>()};
    /* Create Layers */
    layer_list.push_back(std::make_shared<DenseLayer>(train_x[0].size(), 32, true, 0.05, act_list[0]));
    layer_list.push_back(std::make_shared<DenseLayer>(32, 10, true, 0.05, act_list[0]));
    layer_list.push_back(std::make_shared<DenseLayer>(10, 10, false, 0.05, act_list[1]));
    /* Create Model */
    for (auto &l : layer_list) {
        model.addLayer(l);
    }
}

bool MNIST::getLossVector(int y, std::vector<TYPE> &loss) {
    if (!model.getOutput(loss)) return false;
    /* Test Loss */
    //TYPE max_val = *std::max_element(loss.begin(), loss.end());
    //for (int i=0; i<loss.size(); ++i) loss[i] = (loss[i]/max_val);
    ///////////////
    loss[y] -= 1; 
    return true;
}

bool MNIST::train() {
    assertm(train_y.size() == train_x.size(), "train size");
    for (int i=0; i<train_x.size(); ++i) assertm(train_x[i].size() == train_x[0].size(), "train size");
    for (int i=0; i<train_y.size(); ++i) assertm(train_y[i] < 10, "train data");
    bool stat = true;
    std::cout << "Training Started\n";
    for (size_t id=0; id<train_x.size(); ++id) {
        std::vector<TYPE> grad_last(10);
        stat &= model.forwardPass(train_x[id]);
        stat &= getLossVector(train_y[id], grad_last);
        stat &= model.backwardPass(grad_last, id==0);
        for (auto &l : layer_list) {
            l->updateSGD();
        }
        print_progress("Train Progress: ", 1.0*id/(train_x.size()-1));
    }
    std::cout << "\nTraining Done\n";
    return stat;
}

bool MNIST::test() {
    size_t correct_pred = 0;
    int y_pred;
    bool stat = true;
    std::cout << "Testing Started\n";
    std::vector<TYPE> out_pred(10);
    for (size_t id=0; id<test_x.size(); ++id) {
        stat &= model.forwardPass(test_x[id]);
        stat &= model.getOutput(out_pred);
        y_pred = std::distance(out_pred.begin(), std::max_element(out_pred.begin(), out_pred.end()));
        correct_pred += (y_pred == test_y[id]);
        print_progress("Test Progress: ", 1.0*id/(test_x.size()-1));
    }
    std::cout << "\nTesting Done | Accuracy " << 1.0*correct_pred/test_x.size() << "\n";
    TYPE grad_min, grad_max;
#if RECORD
    model.getRecord(grad_min, grad_max);
    std::cout << "Gradient Range: " << grad_min << " to " << grad_max << "\n";
#endif
    return stat;
}
