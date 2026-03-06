#pragma once

//구조
constexpr int N_LAYER = 1;
constexpr int N_EMBD = 16;
constexpr int BLOCK_SIZE = 16;
constexpr int N_HEAD = 4;
constexpr int HEAD_DIM = N_EMBD / N_HEAD;

//optimizer
constexpr double LEARNING_RATE = 0.01;
constexpr double BETA1 = 0.85;
constexpr double BETA2 = 0.99;
constexpr double EPS_ADAM = 1e-8;

//Train
constexpr int NUM_STEPS = 1000;

//Inference
constexpr double TEMPERATURE = 0.5;