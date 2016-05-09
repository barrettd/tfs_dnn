// --------------------------------------------------------------------
//  Dnn.h
//  https://github.com/barrettd/tfs_dnn.git
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef dnn_h
#define dnn_h

#include <vector>


namespace tfs {     // Tree Frog Software

    class DnnLayer;

    class Dnn {
    protected:
        std::vector< DnnLayer* > m_layers;

    public:
        Dnn( void );
        virtual ~Dnn( void );
        
        void clear( void );                                 // Remove all of the layers.
        
        bool addLayerInput( unsigned long xx, unsigned long yy, unsigned long zz = 1 ); // input

        bool addLayerConvolution( void );                   // conv
        bool addLayerDropout( void );                       // dropout
        bool addLayerFullyConnected( void );                // fully connected
        bool addLayerLocalResponseNormalization( void );    // lrn
        bool addLayerMaxout( void );                        // maxout
        bool addLayerPool( void );                          // pool
        bool addLayerRectifiedLinearUnit( void );           // relu
        bool addLayerRegression( void );                    // regression
        bool addLayerSigmoid( void );                       // sigmoid
        bool addLayerSoftmax( void );                       // softmax
        bool addLayerSupportVectorMachine( void );          // svm
        bool addLayerTanh( void );                          // tanh
        
        void randomize( void );                             // Randomize weights and bias.

        bool forward(  void );  // Forward propagate while training
        bool backward( void );  // Back propagate while training
        bool predict(  void );  // Forward progagate when predicting

        bool save( const char *file_path ) const;
        bool load( const char *file_path );
    };
    

    
}   // namespace tfs

#endif /* dnn_h */
