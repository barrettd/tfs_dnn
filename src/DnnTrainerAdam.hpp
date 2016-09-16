//
//  DnnTrainerAdam.hpp
//
//  Created by Barrett Davis on 5/11/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//

#ifndef DnnTrainerAdam_hpp
#define DnnTrainerAdam_hpp

#include "DnnTrainer.hpp"

namespace tfs {
    
    class DnnTrainerAdam : public DnnTrainer {
    protected:
        DNN_NUMERIC m_eps;
        DNN_NUMERIC m_beta1;
        DNN_NUMERIC m_beta2;

    protected:
        virtual DNN_NUMERIC adjustWeights( void );

    public:
        DnnTrainerAdam( Dnn *dnn );
        virtual ~DnnTrainerAdam( void );
        
        DNN_NUMERIC eps( void ) const ;                  // get()
        DNN_NUMERIC eps( const DNN_NUMERIC value );      // set()

        DNN_NUMERIC beta1( void ) const ;                  // get()
        DNN_NUMERIC beta1( const DNN_NUMERIC value );      // set()
        
        DNN_NUMERIC beta2( void ) const ;                  // get()
        DNN_NUMERIC beta2( const DNN_NUMERIC value );      // set()
        
    };
    
}   // namespace tfs

#endif /* DnnTrainerAdam_hpp */
