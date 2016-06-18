// --------------------------------------------------------------------
//  Dnn.h
//  https://github.com/barrettd/tfs_dnn.git
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//  Software released under the The MIT License (MIT)
//  Please feel free to use this software as you wish, but please
//  drop me a line to let me know how you are using this library.
//  Regards, Barrett Davis: barrett (at) thefrog.com
//  http://thefrog.com/barrett/
// --------------------------------------------------------------------
#ifndef dnn_h
#define dnn_h

#include <vector>
#include "Matrix.h"

namespace tfs {     // Tree Frog Software
    
    class DnnLayer;
    class DnnLayerInput;

    class Dnn {
    protected:
        std::vector< DnnLayer* > m_layers;              // This collection "owns" all of the layers
        DnnLayerInput           *m_layer_input;
        DnnLayer                *m_layer_previous;      // Used during creation of the layer stack.
        DnnLayer                *m_layer_output;
        bool                     m_trainable;           // Allocate gradiant arrays if true.
        
    public:
        Dnn( bool trainable = true );
        virtual ~Dnn( void );
        
        bool trainable( void ) const;
        bool trainable( const bool value );
        
        void clear( void );                                 // Remove all of the layers.
        unsigned long count( void ) const;                  // Count of the layers.
        
        bool addLayer( DnnLayerInput *layer );
        bool addLayer( DnnLayer      *layer );

        bool addLayerInput( unsigned long xx, unsigned long yy, unsigned long zz = 1, const bool retain_dw = false ); // input

        bool addLayerConvolution( unsigned long side, unsigned long filters, unsigned long stride = 1, unsigned long pad = 0 ); // conv (square)
        bool addLayerDropout( void );                                       // dropout
        bool addLayerFullyConnected( unsigned long neuronCount );           // fully connected
        bool addLayerLocalResponseNormalization( void );                    // lrn
        bool addLayerMaxout( void );                                        // maxout
        bool addLayerPool( unsigned long side, unsigned long stride = 1, unsigned long pad = 0 );  // pool (square)
        bool addLayerRectifiedLinearUnit( void );                           // relu
        bool addLayerRegression( void );                                    // regression
        bool addLayerSigmoid( void );                                       // sigmoid
        bool addLayerSoftmax( void );                                       // softmax
        bool addLayerSupportVectorMachine( void );                          // svm
        bool addLayerTanh( void );                                          // tanh
        
              DNN_NUMERIC   *getDataInput(    void );
        const DNN_NUMERIC   *getDataInputEnd( void );
              DNN_NUMERIC   *getDataOutput(   void );
        const DNN_NUMERIC   *getDataOutputEnd(void );
        
        Matrix        *getMatrixInput(  void );
        Matrix        *getMatrixOutput( void );
        
        const DnnLayerInput *getLayerInputReadOnly( void ) const;
        DnnLayerInput *getLayerInput(   void );
        DnnLayer      *getLayerOutput(  void );
        
        void initialize( void );                                // Initialize for learning.
        void randomize( void );                                 // Randomize weights and bias.

        void setBias( DNN_NUMERIC value = 0.0 );                // Set biases in all layers.
        bool forward( void );                                   // Forward propagate while training
        bool predict( void );                                   // Forward progagate when predicting
        DNN_NUMERIC backprop( const Matrix     &expectation );  // Back propagate while training, returns loss.
        DNN_NUMERIC backprop( const DNN_INTEGER expectation );  // Back propagate while training, returns loss.
        
        DNN_NUMERIC getCostLoss( void );                        // Forward 
        DNN_NUMERIC getCostLoss( const Matrix     &expectation );
        DNN_NUMERIC getCostLoss( const DNN_INTEGER expectation );
        
    };
    

    
}   // namespace tfs

#endif /* dnn_h */
