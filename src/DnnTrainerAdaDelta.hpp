// -------------------------------------------------------------
//  DnnTrainerAdaDelta.hpp
//
//  Created by Barrett Davis on 5/11/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// -------------------------------------------------------------
// AdaDelta / AdaGrad paper: http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
// -------------------------------------------------------------
#ifndef DnnTrainerAdaDelta_hpp
#define DnnTrainerAdaDelta_hpp

#include "DnnTrainer.hpp"

namespace tfs {
    
    class DnnTrainerAdaDelta : public DnnTrainer {
    protected:
        DNN_NUMERIC m_ro;
        DNN_NUMERIC m_eps;
        
    protected:
        virtual DNN_NUMERIC adjustWeights( void );

    public:
        DnnTrainerAdaDelta( Dnn *dnn );
        virtual ~DnnTrainerAdaDelta( void );
        
        DNN_NUMERIC ro( void ) const ;                  // get()
        DNN_NUMERIC ro( const DNN_NUMERIC value );      // set()
        
        DNN_NUMERIC eps( void ) const ;                  // get()
        DNN_NUMERIC eps( const DNN_NUMERIC value );      // set()
        
        
    };
    
}   // namespace tfs

#endif /* DnnTrainerAdaDelta_hpp */
