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
    /* Create Layers */
    layer_list.push_back(std::make_shared<Linear>(train_x[0].rows(), 100, true, LR));
    layer_list.push_back(std::make_shared<ReLU>(100, 100, false, LR));
    layer_list.push_back(std::make_shared<Linear>(100, 32, true, LR));
    layer_list.push_back(std::make_shared<ReLU>(32, 32, false, LR));
    layer_list.push_back(std::make_shared<Linear>(32, 10, true, LR));
    layer_list.push_back(std::make_shared<Softmax>(10, 10, false, LR));
    /* Create Model */
    for (auto &l : layer_list) {
        model.addLayer(l);
    }
}

bool MNIST::getLossVector(int y, eigen_vec &loss, TYPE &loss_val) {
    if (!model.getOutput(loss)) return false;
    /* Test Loss */
    //TYPE max_val = *std::max_element(loss.begin(), loss.end());
    //for (int i=0; i<loss.size(); ++i) loss[i] = (loss[i]/max_val);
    ///////////////
    // loss[y] -= 1; // What kinda loss is this bruh
    /* log-loss */
    TYPE p = loss[y];
    loss_val = -log(p);
    loss.setZero();
    if (p == 0) loss[y] = LOSS_ZERO;
    else loss[y] = 1/p;
    // std::cout <<  y << "," << loss << " ";
    return true;
}

bool MNIST::train() {
    assertm(train_y.size() == train_x.size(), "train batch size");
    for (int i=0; i<train_x.size(); ++i) 
        assertm(train_x[i].rows() == train_x[0].rows(), "train data size");
    for (int i=0; i<train_y.size(); ++i) 
        assertm(train_y[i] < 10, "train data");
    bool stat = true;
    std::cout << "Training Started\n";
    TYPE loss, norm{0};
    /* Batch Size = 1 */
    for (size_t batch=0; batch<train_x.size()/BATCH_SIZE; ++batch) {
        norm = 1;
        for (size_t id=batch*BATCH_SIZE; id<std::min(train_x.size(), (batch+1)*BATCH_SIZE) && stat; ++id, ++norm) {
            eigen_vec grad_last;
            grad_last.resize(10);
            stat &= model.forwardPass(train_x[id]);
            stat &= getLossVector(train_y[id], grad_last, loss);
            stat &= model.backwardPass(grad_last, id==0); 
            print_progress("Train Progress: ", 1.0*id/(train_x.size()-1));
        }
        for (auto &l : layer_list) {
            l->updateSGD(norm);
        }
    }
    if (stat) std::cout << "\nTraining Done | Loss Value: " << loss << "\n";
    else std::cout << "\nTraining Failed\n";
    return stat;
}

bool MNIST::test() {
    size_t correct_pred = 0;
    int y_pred;
    TYPE max_pred;
    bool stat = true;
    std::cout << "Testing Started\n";
    eigen_vec out_pred;
    out_pred.resize(10);
    for (size_t id=0; id<test_x.size() && stat; ++id) {
        stat &= model.forwardPass(test_x[id]);
        stat &= model.getOutput(out_pred);
        //y_pred = std::distance(out_pred.begin(), std::max_element(out_pred.begin(), out_pred.end()));
        // Find Max Index
        max_pred = out_pred[0];
        y_pred = 0;
        for (int i=1; i<out_pred.rows(); ++i) {
            if (out_pred[i] > max_pred) {
                max_pred = out_pred[i];
                y_pred = i;
            }
        }
        correct_pred += (y_pred == test_y[id]);
        print_progress("Test Progress: ", 1.0*id/(test_x.size()-1));
    }
    if (!stat) {
        std::cout << "\nTesting Failed\n";
        return false;
    }
    std::cout << "\nTesting Done | Accuracy " << 1.0*correct_pred/test_x.size() << "\n";
    TYPE grad_min, grad_max;
#if RECORD
    model.getRecord(grad_min, grad_max);
    std::cout << "Gradient Range: " << grad_min << " to " << grad_max << "\n";
#endif
    return stat;
}
