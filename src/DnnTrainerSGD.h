// --------------------------------------------------------------------
//  DnnTrainerSGD.h
//
//  Created by Barrett Davis on 5/11/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnTrainer.h"

#ifndef DnnTrainerSGD_h
#define DnnTrainerSGD_h


namespace tfs {
    
    class DnnTrainerSGD : public DnnTrainer {
    public:
        DnnTrainerSGD( Dnn *dnn );
        virtual ~DnnTrainerSGD( void );
        
        virtual DNN_NUMERIC train( const Matrix &data, const Matrix &expectation );     // Returns loss.

    };
    
}   // namespace tfs

#endif /* DnnTrainerSGD_h */
