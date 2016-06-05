// --------------------------------------------------------------------
//  Matrix.hpp - 3 or 4 D matrix <template> for DNN_NUMERIC and DNN_INTEGER
//
//  Created by Barrett Davis on 5/10/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------

#ifndef Matrix_h
#define Matrix_h

#include <cmath>        // sqrt()
#include <cstring>      // memset()
#include "Constants.h"
#include "Error.h"
#include "Utility.h"    // random()

namespace tfs {         // Tree Frog Software

    template <typename T> class TMatrix  {      // Possibly use Eigen matricies: https://eigen.tuxfamily.org/
    protected:
        unsigned long m_a;      // x Width
        unsigned long m_b;      // y Height
        unsigned long m_c;      // z Depth
        unsigned long m_d;      // 4th dimension
        unsigned long m_count;  // = x * w * h;  // Count of DNN_NUMERIC elements.
        unsigned long m_length; // = count * sizeof( T )
        T            *m_data;   // Pointer to data[] array:     = &data[0];
        T            *m_end;    // Pointer to end of the array: = &data[m_count];
        
        void allocate( void ) {
            if( m_count > 0 ) {
                m_length = m_count * sizeof( T );   // Length in bytes
                m_data   = new T[m_count];          // = &data[0];          Possibly random values.
                m_end    = m_data + m_count;        // = &m_data[m_count];
            }
        }

    public:
        TMatrix( const TMatrix &other, bool copyOther = false ) :       // Constructor
        m_a( other.m_a ), m_b( other.m_b ), m_c( other.m_c ), m_d( other.m_d ), m_count( other.m_count ),
        m_length( 0 ), m_data( 0 ), m_end( 0 ) {
            allocate();
            if( copyOther ) {
                copy( other );
            }
        }

        TMatrix( const unsigned long aa, const unsigned long bb = 1, const unsigned long cc = 1, const unsigned long dd = 1 ):  // Constructor
        m_a( aa ), m_b( bb ), m_c( cc ), m_d( dd ), m_count( aa * bb * cc * dd ),
        m_length( 0 ), m_data( 0 ), m_end( 0 ) {
            allocate();
        }
        
        ~TMatrix( void ) {      // Destructor
            delete[] m_data;
            m_data   = 0;
            m_end    = 0;
            m_count  = 0;
            m_length = 0;
        }
        
        inline bool ok( void ) const { return m_data != 0 && m_data <= m_end; }
        
        inline unsigned long aa(     void ) const { return m_a; }
        inline unsigned long bb(     void ) const { return m_b; }
        inline unsigned long cc(     void ) const { return m_c; }
        inline unsigned long dd(     void ) const { return m_d; }
        inline unsigned long width(  void ) const { return m_a; }       // aa
        inline unsigned long height( void ) const { return m_b; }       // bb
        inline unsigned long depth(  void ) const { return m_c; }       // cc
        inline unsigned long count(  void ) const { return m_count;  }  // Count of elements.
        inline unsigned long length( void ) const { return m_length; }  // Length in bytes
        
        inline bool isEmpty( void ) const { return m_data == 0 || m_data >= m_end || m_count < 1 || m_length < 1; }
        
        inline       T* data( void ) {               return m_data; }   // Start of data array
        inline const T* end(  void ) const {         return m_end;  }   // End of data array
        inline const T* dataReadOnly( void ) const { return m_data; }   // Start of data array (const)
        
        void randomize( void ) {                // Fill matrix with small positive values
            // Weight normalization is done to equalize the output variance of every neuron,
            // otherwise neurons with a lot of incoming connections will have outputs with a larger variance
            if( isEmpty()) {
                log_error( "empty matrix" );
                return;
            }
            const double  scale = sqrt( 1.0 / m_count );
                  T *       ptr = m_data;
            const T * const end = m_end;
            while( ptr < end ) {
                *ptr++ = (T) randn( scale );
            }
        }

        void zero( void ) {                         // Fill matrix with zeros
            if( m_data != 0 && m_length > 0 ) {
                memset( m_data, 0, m_length );      // Yields IEEE 0 for both integer and real valued variables.
            }
        }
        
        void copy( const TMatrix &matrix ) {        // Copy the contents another matrix.
            if( m_data != 0 && matrix.m_data != 0 && m_length > 0  && m_length == matrix.m_length ) {
                memcpy( m_data, matrix.m_data, m_length );
            }
        }
        
        bool equal( const TMatrix &matrix ) const { // Return true if matricies same dimension and contents.
            if( m_data == 0 || matrix.m_data == 0 || m_length < 1 || m_length != matrix.m_length ||
               m_a != matrix.m_a || m_b != matrix.m_b || m_c != matrix.m_c || m_d != matrix.m_d ) {
                return false;
            }
            return memcmp( m_data, matrix.m_data, m_length ) == 0;
        }
        
        inline unsigned long getIndex( unsigned long aa, unsigned long bb, unsigned long cc, unsigned long dd = 0 ) const {
            return ((( m_a * bb) + aa) * m_c + cc) * m_d + dd;
        }
        
        inline T get( unsigned long aa, unsigned long bb = 0, unsigned long cc = 0, unsigned long dd = 0 ) const {
            const unsigned long index = getIndex( aa, bb, cc, dd );
            return m_data[index];
        }

        inline T set( unsigned long aa, unsigned long bb, unsigned long cc, unsigned long dd, T value ) {
            const unsigned long index = getIndex( aa, bb, cc, dd );
            return m_data[index] = value;
        }

        inline T set( unsigned long x, unsigned long y, unsigned long z, T value ) {
            const unsigned long index = getIndex( x, y, z );
            return m_data[index] = value;
        }

        inline T plusEquals( unsigned long aa, unsigned long bb, unsigned long cc, unsigned long dd, T value ) {
            const unsigned long index = getIndex( aa, bb, cc, dd );
            return m_data[index] += value;
        }

        inline T plusEquals( unsigned long x, unsigned long y, unsigned long z, T value ) {
            const unsigned long index = getIndex( x, y, z );
            return m_data[index] += value;
        }

        inline T dot( const TMatrix &matrix ) const { // Calculate the dot product: scalar = lhs (dot) rhs;
            if( isEmpty()) {
                log_error( "empty matrix" );
                return 0;
            }
            if( matrix.m_count != m_count ) {
                log_error( "Matricies not the same size" );
                return 0;
            }
            const T *       lhs = m_data;
            const T * const end = m_end;
            const T *       rhs = matrix.m_data;
            T result = 0;
            while( lhs < end ) {
                result += *lhs++ * *rhs++;
            }
            return result;
        }
        
        inline T min( void ) const {                // Return minimum value.
            if( isEmpty()) {
                log_error( "empty matrix" );
                return 0;
            }
            const T *      data = m_data;
            const T * const end = m_end;
            
            T result = *data++;
            while( data < end ) {
                if( *data < result ) {
                    result = *data;
                }
                data++;
            }
            return result;
        }
        
        inline T max( void ) const {                // Return maximum value.
            if( isEmpty()) {
                log_error( "empty matrix" );
                return 0;
            }
            const T *      data = m_data;
            const T * const end = m_end;
            
            T result = *data++;
            while( data < end ) {
                if( *data > result ) {
                    result = *data;
                }
                data++;
            }
            return result;
        }
 
        inline T sum( void ) const {                // Return sum of matrix elements.
            if( isEmpty()) {
                log_error( "empty matrix" );
                return 0;
            }
            const T *      data = m_data;
            const T * const end = m_end;
            
            T result = *data++;
            while( data < end ) {
                result += *data++;
            }
            return result;
        }
        
        inline void add( T value ) const {          // Add a scalar to each element.
            if( isEmpty()) {
                log_error( "empty matrix" );
                return ;
            }
            const T *      data = m_data;
            const T * const end = m_end;
            
            while( data < end ) {
                *data++ += value;
            }
            return;
        }


        inline void multiply( T value ) const {     // Multiple each element by a scalar.
            if( isEmpty()) {
                log_error( "empty matrix" );
                return ;
            }
            const T *      data = m_data;
            const T * const end = m_end;
            
            while( data < end ) {
                *data++ *= value;
            }
            return;
        }
        
    };  // class TMatrix
    
    template <typename T> struct TTrainable {       // Tuple of weight Matrix and gradiant Matrix.
              T *weightStart;                       // Convienience pointers to the start and end of data arrays
        const T *weightEnd;
              T *gradiantStart;
        const T *gradiantEnd;
              T  l1_decay_mul;
              T  l2_decay_mul;

        TTrainable( void ):
        weightStart( 0 ), weightEnd( 0 ), gradiantStart( 0 ), gradiantEnd( 0 ),
        l1_decay_mul( 1.0 ), l2_decay_mul( 1.0 ) {
        }
        
        TTrainable( TMatrix< T > *weights, TMatrix< T > *gradiants ) :
        weightStart( 0 ), weightEnd( 0 ), gradiantStart( 0 ), gradiantEnd( 0 ),
        l1_decay_mul( 1.0 ), l2_decay_mul( 1.0 ) {
            setWeight(   weights   );
            setGradiant( gradiants );
        }
        
        ~TTrainable( void ) {
            weightStart   = 0;
            weightEnd     = 0;
            gradiantStart = 0;
            gradiantEnd   = 0;
        }
        
        inline bool ok( void ) const {
            return weightStart != 0 && weightStart < weightEnd && gradiantStart != 0 && gradiantStart < gradiantEnd;
        }
        
        void setWeight( TMatrix< T > *matrix ) {
            if( matrix != 0 ) {
                weightStart = matrix->data();
                weightEnd   = matrix->end();
            } else {
                weightStart = 0;
                weightEnd   = 0;
            }
        }

        void setGradiant( TMatrix< T > *matrix ) {
            if( matrix != 0 ) {
                gradiantStart = matrix->data();
                gradiantEnd   = matrix->end();
            } else {
                gradiantStart = 0;
                gradiantEnd   = 0;
            }
        }

    };  // struct TTrainable
    
    typedef TMatrix< DNN_NUMERIC >  Matrix;     // Typically double
    typedef TMatrix< DNN_INTEGER > DMatrix;     // Typically long
    
    typedef TTrainable< DNN_NUMERIC >  Trainable;
    typedef TTrainable< DNN_INTEGER > DTrainable;
    
    inline bool matrixBad( const  Matrix * const matrix ) { return matrix == 0 || !matrix->ok(); }
    inline bool matrixBad( const DMatrix * const matrix ) { return matrix == 0 || !matrix->ok(); }
    
}   // namespace tfs

#endif /* Matrix_h */
