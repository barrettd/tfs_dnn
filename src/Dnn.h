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
// --------------------------------------------------------------------
#ifndef dnn_h
#define dnn_h

#include <vector>


namespace tfs {     // Tree Frog Software

    class DnnLayer;
    class DnnLayerInput;

    class Dnn {
    protected:
        std::vector< DnnLayer* > m_layers;              // This collection "owns" all of the layers
        DnnLayerInput           *m_layer_input;
        DnnLayer                *m_layer_output;
        
    protected:
        bool addLayer( DnnLayerInput *layer );
        bool addLayer( DnnLayer      *layer );

    public:
        Dnn( void );
        virtual ~Dnn( void );
        
        void clear( void );                                 // Remove all of the layers.
        unsigned long count( void ) const;                  // Count of the layers.
        
        bool addLayerInput( unsigned long xx, unsigned long yy, unsigned long zz = 1 ); // input

        bool addLayerConvolution( unsigned long side, unsigned long filters, unsigned long stride = 1, unsigned long pad = 0 ); // conv (square)
        bool addLayerDropout( void );                       // dropout
        bool addLayerFullyConnected( unsigned long xx, unsigned long yy, unsigned long zz = 1 );  // fully connected
        bool addLayerLocalResponseNormalization( void );    // lrn
        bool addLayerMaxout( void );                        // maxout
        bool addLayerPool( unsigned long side, unsigned long stride = 1 );  // pool (square)
        bool addLayerRectifiedLinearUnit( void );           // relu
        bool addLayerRegression( void );                    // regression
        bool addLayerSigmoid( void );                       // sigmoid
        bool addLayerSoftmax( unsigned long classCount );   // softmax
        bool addLayerSupportVectorMachine( void );          // svm
        bool addLayerTanh( void );                          // tanh
        
        DnnLayerInput *getLayerInput(  void );
        DnnLayer      *getLayerOutput( void );
        
        void randomize( void );                             // Randomize weights and bias.

        bool forward(  void );  // Forward propagate while training
        bool backprop( void );  // Back propagate while training
        bool predict(  void );  // Forward progagate when predicting

        bool save( const char *file_path ) const;
        bool load( const char *file_path );
    };
    

    
}   // namespace tfs

#endif /* dnn_h */
