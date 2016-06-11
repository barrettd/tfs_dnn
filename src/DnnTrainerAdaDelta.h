//
//  DnnTrainerAdaDelta.h
//
//  Created by Barrett Davis on 5/11/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#ifndef DnnTrainerAdaDelta_h
#define DnnTrainerAdaDelta_h

#include "DnnTrainer.h"

namespace tfs {
    
    class DnnTrainerAdaDelta : public DnnTrainer {
    protected:
        DNN_NUMERIC m_ro;
        DNN_NUMERIC m_eps;
        
    public:
        DnnTrainerAdaDelta( Dnn *dnn );
        virtual ~DnnTrainerAdaDelta( void );
        
        virtual DNN_NUMERIC train( const DNN_INTEGER expectation );     // Returns loss.
        
        DNN_NUMERIC ro( void ) const ;                  // get()
        DNN_NUMERIC ro( const DNN_NUMERIC value );      // set()
        
        DNN_NUMERIC eps( void ) const ;                  // get()
        DNN_NUMERIC eps( const DNN_NUMERIC value );      // set()
        
        
    };
    
}   // namespace tfs

//this.ro = typeof options.ro !== 'undefined' ? options.ro : 0.95; // used in adadelta
//this.eps = typeof options.eps !== 'undefined' ? options.eps : 1e-8; // used in adam or adadelta
// this.xsum = []; // used in adam or adadelta

#endif /* DnnTrainerAdaDelta_h */
