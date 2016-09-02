// --------------------------------------------------------------------
//  DnnLayer.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef dnnLayer_h
#define dnnLayer_h

#include "Matrix.h"

namespace tfs {
    
    enum LayerType {
        LAYER_INPUT = 0,                        // input layer
        LAYER_CONVOLUTION,                      // conv
        LAYER_DROPOUT,                          // dropout
        LAYER_FULLY_CONNECTED,                  // fc
        LAYER_LOCAL_RESPONSE_NORMALIZATION,     // lrn
        LAYER_MAXOUT,                           // maxout
        LAYER_POOL,                             // pool
        LAYER_RECTIFIED_LINEAR_UNIT,            // relu
        LAYER_REGRESSION,                       // regression
        LAYER_SIGMOID,                          // sigmoid
        LAYER_SOFTMAX,                          // softmax
        LAYER_SUPPORT_VECTOR_MACHINE,           // svm
        LAYER_TANH,                             // tanh
        
        LAYER_COUNT                             // This is not a layer type, but a max value used when range checking.
    };
    
    class DnnLayer {                    // Base class of all layers.
    protected:
        const LayerType m_layer_type;   // Used in serialization (low impact RTTI)
        const Matrix   *m_in_a;         // Input:    Activations of previous layer, if any.
              Matrix   *m_in_dw;        // Input:    dw of previous layer, if any.
        Matrix         *m_w;            // Internal: Weights, to act on input activations from previous layer
        Matrix         *m_dw;           // Internal: Weight derivative, will be null when not training.
        Matrix         *m_bias_w;       // Internal: Bias, to act on input activations from previous layer
        Matrix         *m_bias_dw;      // Internal: Bias derivative, will be null when not training.
        Matrix         *m_out_a;        // Output:   Activations, output of a neuron.
        Matrix         *m_out_dw;       // Output:   Weight derivative, will be null when not training.
        DNN_NUMERIC     m_l1_decay_mul;
        DNN_NUMERIC     m_l2_decay_mul;

        DnnLayer       *m_prev_layer;
        DnnLayer       *m_next_layer;
        
        void setup( const bool trainable = true );
        void teardown( void );
        
        
    public:
        DnnLayer( const LayerType layerType );
        DnnLayer( const LayerType layerType, DnnLayer *previousLayer );
        virtual ~DnnLayer( void );
        
        // Layer attributes:
        LayerType  layerType( void ) const;
        Matrix     *outA(  void );              // Output Neuron Activations
        Matrix     *outDw( void );              // d/dw Output Neuron Activations
        Matrix     *weights(  void );           // Internal Neuron connection weights   (w)
        Matrix     *gradient( void );           // Internal Neuron connection gradient  (dw)
        Matrix     *bias(   void );             // Internal Neuron connection bias      (bias.w)
        Matrix     *biasDw( void );             // Internal Neuron connection bias dw   (bias.dw)

        const Matrix *outAReadOnly(    void ) const;              // Output Neuron Activations
        const Matrix *outDwReadOnly(    void ) const;              // d/dw Output Neuron Activations
        const Matrix *weightsReadOnly(  void ) const; // Internal Neuron connection weights   (w)
        const Matrix *gradientReadOnly( void ) const; // Internal Neuron connection gradient  (dw)
        const Matrix *biasReadOnly(     void ) const; // Internal Neuron connection bias      (bias.w)
        const Matrix *biasDwReadOnly(   void ) const; // Internal Neuron connection bias dw   (bias.dw)

        DNN_NUMERIC l1DecayMultiplier( void ) const;
        DNN_NUMERIC l1DecayMultiplier( DNN_NUMERIC value );
        
        DNN_NUMERIC l2DecayMultiplier( void ) const;
        DNN_NUMERIC l2DecayMultiplier( DNN_NUMERIC value );
        
        // Linked list pointers:
        DnnLayer *getPreviousLayer( void ) const;
        DnnLayer *setPreviousLayer( DnnLayer *layer );
        DnnLayer *getNextLayer( void ) const;
        DnnLayer *setNextLayer( DnnLayer *layer );
        
        // ----------------------------------------------------------------------------------------------
        // These methods perform a function, then call the next layer in the chain to do the same.
        // ----------------------------------------------------------------------------------------------
        virtual void initialize( void );        // Zero activations, gradient and randomize weights. Forward calling.
        virtual void randomize(  void );        // Randomize gradient. Forward calling.
        
        // Perform function in just the current layer
        virtual bool runBias( DNN_NUMERIC value );              // Set biases in current layer.
        virtual bool runForward(  void );                       // Forward propagate while training
        virtual bool runPredict(  void );                       // Forward progagate when predicting
        virtual bool runBackprop( void );                       // Back propagate while training
        virtual DNN_NUMERIC runBackprop( const  Matrix    &expectation );   // Back propagate while training (Last layer)
        virtual DNN_NUMERIC runBackprop( const DNN_INTEGER expectation );   // Back propagate while training (Last layer)
        
        // Perform function in all layers (either forward or backward)
        void setBias( DNN_NUMERIC value = 0.0 );                // Set biases in all layers.
        bool forward(  void );                                  // Forward propagate while training
        bool predict(  void );                                  // Forward progagate when predicting
        bool backprop( void );                                  // Back propagate while training
        DNN_NUMERIC backprop( const  Matrix    &expectation );  // Last layer
        DNN_NUMERIC backprop( const DNN_INTEGER expectation );  // Last layer
        
        
    };
    
    
}   // namespace tfs

#endif /* dnnLayers_h */
